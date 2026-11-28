"""Pinned resource downloads through configured workbench devices, without shell recipes."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

def atomic(path, data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    tmp.replace(path)

def validate(recipe):
    target, root, python=(Path(recipe[k]) for k in ('target','root','python'))
    if not all(p.is_absolute() and '..' not in p.parts for p in (target,root,python)):
        raise ValueError('配置必须使用绝对路径')
    if target.parent!=root or len(root.parts)<3 or target.name.startswith('.'):
        raise ValueError('目标必须为配置资源根目录的直接子目录')
    if recipe['kind'] not in ('dataset','weights') or recipe['repo_type'] not in ('model','dataset'):
        raise ValueError('不支持的资源类型')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',recipe['repo']) or not re.fullmatch(r'[a-f0-9]{40}',recipe['revision']):
        raise ValueError('必须使用固定 HF commit')
    names=set()
    for row in recipe['files']:
        name=Path(row['path'])
        if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0] in ('.cache','.workbench-trash','.workbench-downloads') or name in names:
            raise ValueError('不安全或重复的清单路径')
        names.add(name)
        if type(row['bytes']) is not int or row['bytes']<0 or not (re.fullmatch(r'[a-f0-9]{64}',row.get('sha256') or '') or re.fullmatch(r'[a-f0-9]{40}',row.get('git_blob_sha1') or '')):
            raise ValueError('每个文件必须登记大小和摘要')
    if not names:
        raise ValueError('空清单')
    return sum(row['bytes'] for row in recipe['files'])

def verify(path,row):
    if path.is_symlink() or not path.is_file() or path.stat().st_size!=row['bytes']:
        raise ValueError('文件大小或类型不符: '+row['path'])
    digest=hashlib.sha256() if row.get('sha256') else hashlib.sha1()
    if not row.get('sha256'):
        digest.update(('blob '+str(row['bytes'])+'\0').encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(chunk)
    if digest.hexdigest()!=(row.get('sha256') or row.get('git_blob_sha1')):
        raise ValueError('摘要不符: '+row['path'])

def paths(recipe):
    target=Path(recipe['target'])
    work=target.parent/'.workbench-downloads'/target.name
    identity=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()[:20]
    return target,work,identity

def read_state(work):
    try:
        return json.loads((work/'state.json').read_text())
    except FileNotFoundError:
        return {}

def active(work):
    if not (work/'lock').is_file():
        return False
    with (work/'lock').open('r') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False

def operation(recipe,action,source=None):
    total=validate(recipe)
    target,work,identity=paths(recipe)
    if target.parent.resolve()!=target.parent or work.resolve()!=work or target.is_symlink():
        raise ValueError('资源根目录或工作目录不可通过软链接')
    state=read_state(work)
    if state and state.get('job_id')!=identity:
        raise ValueError('此目标已绑定不同配方，不能覆盖')
    running=active(work)
    if action=='status':
        status='running' if running else state.get('status','not_started')
        if not running and (status in ('starting','running') or status=='complete' and not target.is_dir()):
            status='interrupted'
        return {**state,'status':status,'job_id':identity,'target':str(target),'total_bytes':total}
    if action not in ('preview','start'):
        raise ValueError('未知下载操作')
    if not target.parent.is_dir():
        raise ValueError('设备资源根目录不存在')
    if running:
        raise ValueError('同目标已有下载，请查询状态')
    if os.path.lexists(target):
        raise ValueError('目标已存在，请核对并登记，禁止覆盖')
    check=subprocess.run([recipe['python'],'-c','import huggingface_hub'],capture_output=True,timeout=20)
    if check.returncode:
        raise ValueError('配置的 Python 环境无法导入 huggingface_hub')
    free=shutil.disk_usage(target.parent).free
    if free<total+1024**3:
        raise ValueError('可用空间不足：完整清单+1 GiB 预留')
    plan=dict(job_id=identity,target=str(target),total_bytes=total,free_bytes=free,python=recipe['python'],
              repo=recipe['repo'],revision=recipe['revision'],file_count=len(recipe['files']))
    if action=='preview':
        return plan
    work.mkdir(parents=True,exist_ok=True)
    with (work/'lock').open('a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('同目标已有下载')
        if os.path.lexists(target) or (read_state(work) and read_state(work).get('job_id')!=identity):
            raise ValueError('预检后目标发生变化')
        atomic(work/'recipe.json',recipe)
        atomic(work/'state.json',{**plan,'status':'starting','verified_bytes':0,'updated_at':time.time()})
        with (work/'worker.log').open('a') as log:
            child=subprocess.Popen([recipe['python'],'-c',source,'--worker',str(work/'recipe.json'),str(lock.fileno())],
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,pass_fds=(lock.fileno(),))
    return {**plan,'status':'starting','pid':child.pid}

def worker(recipe_path,lock_fd):
    recipe=json.loads(Path(recipe_path).read_text())
    total=validate(recipe)
    target,work,identity=paths(recipe)
    stage=work/'staging'
    state=dict(job_id=identity,target=str(target),total_bytes=total,verified_bytes=0,
               verified_files=0,status='running',pid=os.getpid())
    def save(**updates):
        state.update(updates,updated_at=time.time())
        atomic(work/'state.json',state)
    try:
        from huggingface_hub import hf_hub_download
        stage.mkdir(exist_ok=True)
        if stage.resolve()!=stage:
            raise ValueError('不安全的暂存路径')
        save()
        for row in recipe['files']:
            name=row['path']
            path=stage/name
            if path.resolve()!=path:
                raise ValueError('不安全的文件路径')
            save(current_file=name)
            if path.exists():
                try:
                    verify(path,row)
                except ValueError:
                    path.unlink()
            if not path.exists():
                hf_hub_download(recipe['repo'],name,repo_type=recipe['repo_type'],
                                revision=recipe['revision'],local_dir=str(stage))
                verify(path,row)
            state['verified_bytes']+=row['bytes']
            state['verified_files']+=1
            save()
        if os.path.lexists(target):
            raise ValueError('目标已出现，拒绝覆盖')
        stage.rename(target)
        save(status='complete',current_file=None)
    except Exception as error:
        save(status='failed',error=type(error).__name__+': '+str(error)[:800])
        raise
    finally:
        os.close(lock_fd)

class DownloadService:
    def __init__(self,project,devices):
        self.project=Path(project)
        self.devices=devices
    def recipes(self):
        path=self.project/'configs/workbench_downloads.json'
        if not path.is_file():
            return {}
        result={}
        for item in json.loads(path.read_text()).get('recipes',[]):
            key=item['id']
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',key) or key in result:
                raise ValueError('配方 ID 非法或重复')
            manifest=(self.project/item['manifest']).resolve()
            if not manifest.is_relative_to(self.project.resolve()) or not manifest.is_file():
                raise ValueError('清单必须在仓库内')
            resource=next((r for r in json.loads(manifest.read_text())['resources'] if r['name']==item['resource']),None)
            if resource is None:
                raise ValueError('清单中缺少该资源')
            recipe={k:item[k] for k in ('id','host','kind','root','target','python')}
            recipe.update({k:resource[k] for k in ('repo','repo_type','revision','files')})
            validate(recipe)
            result[key]=recipe
        return result
    def request(self,action,recipe_id=None):
        recipes=self.recipes()
        if action=='options':
            return {'recipes':[{**{k:v for k,v in r.items() if k!='files'},'total_bytes':sum(f['bytes'] for f in r['files'])} for r in recipes.values()]}
        if action not in ('status','preview','start') or recipe_id not in recipes:
            raise ValueError('请选择登记的下载配方')
        recipe=recipes[recipe_id]
        source=Path(__file__).read_text()
        host=recipe['host']
        if host=='4090':
            return operation(recipe,action,source)
        device=self.devices.get({'A100':'a100','A6000':'a6000'}.get(host,host))
        if not device or not device.get('host'):
            raise ValueError('设备尚未配置')
        reply=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',device['host'],
            'python3 -c '+shlex.quote(source)],input=json.dumps(dict(recipe=recipe,action=action,source=source)),
            text=True,capture_output=True,timeout=45)
        if reply.returncode:
            raise ValueError('目标设备不可达: '+reply.stderr[-500:])
        result=json.loads(reply.stdout)
        if result.get('error'):
            raise ValueError(result['error'])
        return result

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        worker(sys.argv[2],int(sys.argv[3]))
    else:
        try:
            print(json.dumps(operation(**json.load(sys.stdin)),ensure_ascii=False))
        except (OSError,ValueError,subprocess.SubprocessError,KeyError) as error:
            print(json.dumps({'error':str(error)},ensure_ascii=False))
