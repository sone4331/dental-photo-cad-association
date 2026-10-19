"""Cost-only sample: original42 renderer, isolated output, no new accuracy."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import csv,json,sys,time,traceback
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PUB=HERE.parent
sys.path.insert(0,str(PUB))
def main():
    p=json.loads((HERE/'offline_cost_protocol.json').read_text());paths=[ROOT/'dataset/models'/f'{m}.stl' for m in p['model_ids']]
    missing=[m for m,path in zip(p['model_ids'],paths) if not path.is_file()]
    if missing:
        (HERE/'OFFLINE_RENDER_COST.json').write_text(json.dumps(dict(status='unavailable_raw_CADs_on_server',missing_ids=missing,render_time_not_estimated=True),indent=2));return
    import render_gallery as renderer
    config=json.loads((PUB/'cache/renders_sphere42/camera_config.json').read_text())
    renderer.ANGLES=tuple(tuple(x) for x in config['poses']);assert len(renderer.ANGLES)==42
    result=[];r=None
    try:
        r=renderer.pyrender.OffscreenRenderer(192,192)
        renderer.render_one(r,paths[0],HERE/'offline_cost_renders/warmup')
        from OpenGL import GL
        gl_info={}
        for name,key in [('vendor',GL.GL_VENDOR),('renderer',GL.GL_RENDERER),('version',GL.GL_VERSION)]:
            value=GL.glGetString(key);gl_info[name]=value.decode('utf-8','replace') if value else 'unavailable'
        for mid,path in zip(p['model_ids'],paths):
            t=time.monotonic();info=renderer.render_one(r,path,HERE/'offline_cost_renders'/mid);elapsed=time.monotonic()-t
            files=list((HERE/'offline_cost_renders'/mid).glob('view_*.png'));assert len(files)==42
            result.append(dict(model_id=mid,STL_bytes=path.stat().st_size,vertices=info['vertices'],views=42,seconds=elapsed,PNG_bytes=sum(f.stat().st_size for f in files)))
        with (HERE/'offline_render_observations.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
        vv=[x['seconds'] for x in result]
        out=dict(status='complete16_CAD_cost_sample',CADs=16,frames=672,warmup_CADs=1,mean_seconds_per_CAD=float(np.mean(vv)),median_seconds_per_CAD=float(np.median(vv)),
                 estimate438_seconds=float(np.mean(vv)*438),estimate_is_not_measured_full_catalog_time=True,
                 measured_includes='STL read/normalization, scene,42renders,PNG encoding/write',excludes='OpenGL initialization and feature extraction',
                 OpenGL=gl_info,
                 recognition_gallery_unchanged=True)
    except Exception:
        out=dict(status='render_cost_unavailable',completed_CADs=len(result),error=traceback.format_exc(),recognition_gallery_unchanged=True)
    finally:
        if r is not None:r.delete()
    (HERE/'OFFLINE_RENDER_COST.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
