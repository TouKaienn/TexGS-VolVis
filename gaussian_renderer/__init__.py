from gaussian_renderer.disk_render import disk_render
from gaussian_renderer.texGS_render_wLight import texGS_render_wLight
from gaussian_renderer.texGS_render_woLight import texGS_render_woLight
from gaussian_renderer.stylize_render import stylize_render
from gaussian_renderer.stylize_render_inf import stylize_render_inf
render_fn_dict = {
    "2DGS": disk_render,
    "TexGS": texGS_render_wLight,
    "TexGS_noLight": texGS_render_woLight,
    "stylize": stylize_render,
    "stylize_inf": stylize_render_inf
}