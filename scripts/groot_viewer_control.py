"""Pause only the unused viewer after camera graph initialization and RGB resizing."""


def pause_viewer(viewer):
    # Keep camera prims, annotators and graph names intact. Early omission of the
    # viewer exposes a SyntheticData node-lifecycle failure in robot RGB resizing.
    viewer._render_product.hydra_texture.set_updates_enabled(False)
    viewer._viewport.visible = False
    viewer._viewport.viewport_api.updates_enabled = False
    return [str(viewer._render_product.path), str(viewer._viewport.viewport_api.render_product_path)]


def pause_eval_viewer():
    import omnigibson as og
    from groot_eval_timing import trace_event

    viewer = og.sim.viewer_camera
    paths = pause_viewer(viewer)
    trace_event('rollout.pause_unused_viewer', 'end', products=','.join(paths))
