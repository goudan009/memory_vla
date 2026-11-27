"""Exercise continuation gates without allocating any model/simulator job."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import native_supervisor as watch


def plan():
    return dict(mode='dual2',preset='official-eval',task_id=40,base_seed=0,
        seed_mode='sequence',episodes_per_gpu=5,repetitions=1,n_envs_per_gpu=2,
        action_chunk=8,record_video=True,video_steps_per_render=24,video_fps=20,
        run_tag=watch.TAG,version='V_2_1',group='baseline',
        instances={'0':list(range(5)),'1':list(range(5,10))},runner_max_episode_steps=999999999,
        policy_identity=dict(checkpoint=watch.CHECKPOINT,source_dir=watch.SOURCE,
            source_digest=watch.DIGEST,training_id=watch.RUN,binding='observed_process'))


class NativeContinuationTests(unittest.TestCase):
    def test_dual2_request_preserves_original_budget(self):
        request=watch.evaluation_request()
        self.assertEqual((request['mode'],request['episodes_per_gpu'],request['repetitions']),('dual2',5,1))
        self.assertEqual(request['instances_gpu0']+request['instances_gpu1'],list(range(10)))
        self.assertEqual((request['action_steps'],request['preset']),(8,'official-eval'))

    def test_wrong_checkpoint_and_single_gpu_cannot_pass(self):
        watch.check_plan(plan())
        bad=plan();bad['mode']='single1'
        with self.assertRaises(ValueError):watch.check_plan(bad)
        bad=plan();bad['policy_identity']['checkpoint']='old-weight'
        with self.assertRaises(ValueError):watch.check_plan(bad)

    def test_ten_native_episodes_and_videos_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            (output/'results.json').write_text(json.dumps(dict(configuration=plan())))
            rows=[]
            for gpu in (0,1):
                for env,ids in [(0,range(gpu*5,gpu*5+2)),(1,range(gpu*5+2,gpu*5+5))]:
                    for number,instance in enumerate(ids):
                        name=f'{gpu}_{env}_{number}.mp4';(output/name).write_bytes(b'closed-video')
                        rows.append(dict(gpu_index=gpu,env_idx=env,instance_id=instance,seed=env+number,
                            valid=True,evaluation_mode='official-eval',task_id=40,terminated=False,
                            truncated=True,q_score=0.,task_progress=0.,video_file=name,
                            native_termination=dict(success=False,termination_conditions={'timeout':{'done':True}})))
            def write_rows():
                (output/'episodes.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
            write_rows();self.assertEqual(watch.validate_result(output)['valid_episodes'],10)
            rows[0]['valid']=False;write_rows()
            with self.assertRaises(ValueError):watch.validate_result(output)
            rows[0]['valid']=True;rows[0]['native_termination']['termination_conditions']['timeout']['done']=False
            write_rows()
            with self.assertRaises(ValueError):watch.validate_result(output)

    def test_running_and_paused_training_do_not_launch_evaluation(self):
        for status in ('training','paused'):
            detail=dict(config=dict(recipe='native_random_v2',run_id=watch.RUN,
                source_manifest={'digest':watch.DIGEST},steps=6000,gpus=[0,1,2,3]),
                state=dict(status=status,step=100),metrics=[])
            def api(method,route,body=None):
                self.assertEqual(method,'GET')
                return detail if route.startswith('/api/training/detail') else {}
            state={'status':'training'}
            with patch.object(watch,'api',side_effect=api),patch.object(watch,'write'):
                watch.tick(state)
            self.assertEqual(state['status'],'training' if status=='training' else 'user_paused')

    def test_uncertain_submission_is_not_retried(self):
        state={'actions':{'start-evaluation':{'status':'sending'}}}
        with patch.object(watch,'api') as api:
            with self.assertRaises(RuntimeError):watch.mutate(state,'start-evaluation','/api/launch/start',{})
            api.assert_not_called()

    def test_ready_exact_service_launches_only_locked_eval(self):
        service=dict(pid=123,start_ticks=456,state='ready',port=watch.PORT,
            checkpoint=watch.CHECKPOINT,source_dir=watch.SOURCE,policy_seed=42,
            gpu_index=0,history_mode='normal',command='serve --inference-steps 4')
        calls=[]
        def api(method,route,body=None):
            calls.append((method,route,body))
            if route=='/api/launch/jobs':return {'jobs':[]}
            if route.startswith('/api/inference/jobs'):return {'jobs':[service]}
            if route=='/api/launch/preview':return {'plan':plan()}
            if route=='/api/launch/start':return {'id':'actual-workbench-job'}
            raise AssertionError(route)
        state={}
        with tempfile.TemporaryDirectory() as directory,patch.object(watch,'ROOT',Path(directory)),\
             patch.object(watch,'api',side_effect=api),patch.object(watch,'write'),\
             patch.object(watch,'save'),patch.object(watch,'observe_command',side_effect=lambda job:job):
            watch.eval_tick(state)
        mutations=[(route,body) for method,route,body in calls if method=='POST']
        self.assertEqual([route for route,body in mutations],['/api/launch/preview','/api/launch/start'])
        self.assertEqual(mutations[0][1],mutations[1][1])
        self.assertEqual(state['evaluation_job']['id'],'actual-workbench-job')

    def test_inference_steps_mismatch_cannot_be_reused(self):
        service=dict(checkpoint=watch.CHECKPOINT,source_dir=watch.SOURCE,policy_seed=42,
            gpu_index=0,history_mode='normal',port=watch.PORT,command='serve --inference-steps 16')
        self.assertFalse(watch.matches_service(service))


if __name__=='__main__':unittest.main()
