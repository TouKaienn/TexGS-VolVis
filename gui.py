from icecream import install
install()
ic.configureOutput(includeContext=True) #type: ignore


import os
import torchvision.transforms
import dearpygui.dearpygui as dpg
from dearpygui_ext.themes import create_theme_imgui_light

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from gaussian_renderer import render_fn_dict
from scene import GaussianModel
from utils.general_utils import safe_state
from utils.camera_utils import Camera
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams

from utils.graphics_utils import focal2fov,fov2focal
from scene.palette_color import LearningPaletteColor
from scene.opacity_trans import LearningOpacityTransform
from scene.light_trans import LearningLightTransform

import cv2

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor, #type: ignore
                              sam_model_registry) 
from utils.segment_utils import *
from utils.gui_utils import *
from utils.dpge_movable_group import MovableGroup as movable_group
import copy
DEFAULT_FD_DIR = None
SAM_CKPT_PATH = "/home/dullpigeon/Desktop/Spring/InstructGS2GS-Texture2DGS/cache/sam_ckpt/sam_vit_h_4b8939.pth"


class GUI:
    def __init__(self, args, render_fn, render_kwargs, dataset_kwargs, TFnums, gs_list_kwargs):
        self.ctrlW = 475
        self.widget_indent = 75
        self.widget_top = 150
        self.imgW = 800
        self.imgH = 800
        
        c2w = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.525, 0.85, 3.43],
            [0.0, -0.85, 0.525, 2.12],
            [0.0, 0.0, 0.0, 1.0]
        ])
        c2w /= 2
        c2w[:3, 1:3] *= -1
        
        self.debug = False 
        center = np.zeros(3)
        rot = c2w[:3, :3]
        translate = c2w[:3, 3] - center
        self.TFnums = TFnums
        self.render_fn = render_fn
        self.render_kwargs = render_kwargs
        self.original_palette_colors = [torch.clamp(copy.deepcopy(self.render_kwargs["transform_dict"]["palette_colors"][TFidx].palette_color.detach()), min=0.0, max=1.0) for TFidx in range(TFnums)]
        self.original_pc = copy.deepcopy(self.render_kwargs["pc"])
        fovx = 30 * np.pi / 180
        fovy = focal2fov(fov2focal(fovx, self.imgW), self.imgH)
        self.cam = ArcBallCamera(self.imgW, self.imgH, fovy=fovy * 180 / np.pi, rot=rot, translate=translate, center=center)

        self.render_buffer = np.zeros((self.imgW, self.imgH, 3), dtype=np.float32)
        self.resize_fn = torchvision.transforms.Resize((self.imgH, self.imgW), antialias=True)
        self.downsample = 1
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        self.prev_mouseX = None
        self.prev_mouseY = None
        self.is_segmenting = False
        
        self.light_elevation = 0
        self.light_angle = 180
        self.useHeadlight = True
        
        self.menu = None
        self.mode = "render"
        
        self.focus_TF_idx = 0

        sam = sam_model_registry['vit_h'](checkpoint=SAM_CKPT_PATH).to("cuda")
        self.sam_predictor = SamPredictor(sam)
        self.gt_render_mask = torch.zeros(1, self.imgH, self.imgW, dtype=torch.float32, device="cuda")
        self.ensemble_cameras = get_ensemble_cameras(args, dataset_kwargs["ensemble_json_views_path"])
        self.intermediate_render = None
        self.input_points = []
        self.input_label = []
        self.temp_save_kwargs = gs_list_kwargs

        
        self.step()
        
        dpg.create_context()
        
        self.setup_font_theme()
        
        
        light_theme = create_theme_imgui_light()
        dpg.bind_theme(light_theme)
        self.register_dpg()

    def __del__(self):
        dpg.destroy_context()
    
    def _init_file_dialog(self):
        def callback_select_texture(sender, app_data):
            print(app_data['file_path_name'])
            print(self.focus_TF_idx)
            num_GSs_TFs = self.render_kwargs['pc'].get_num_GSs_TFs
            texture_list, mappings_list = self.temp_save_kwargs['texture_list'], self.temp_save_kwargs['mappings_list']
            texture_list[self.focus_TF_idx], mappings_list[self.focus_TF_idx] = load_compose_texture(app_data['file_path_name'])
            texture = JaggedTexture.create_from_textures(texture_list,num_GSs_TFs)
            self.render_kwargs['pc'].load_jagged_texture(texture, mappings_list)
            self.render_kwargs['transform_dict']['palette_colors'][self.focus_TF_idx].set_black()
            self._sync_opacity_color()
            
            
        with dpg.file_dialog(directory_selector=False, show=False, width=800, height=500, default_path=DEFAULT_FD_DIR, callback=callback_select_texture, tag="_file_dialog_id"):
            dpg.add_file_extension(".npz", color=(0, 0, 0, 255))
    
    def _sync_opacity_color(self):
        for TFidx in range(self.TFnums):
            dpg.set_value(f"_slider_TF{TFidx+1}", self.render_kwargs["transform_dict"]["opacity_factors"][TFidx].opacity_factor.item())
            color_value = [int(x*255) for x in self.render_kwargs["transform_dict"]["palette_colors"][TFidx].palette_color.detach().cpu().numpy()]
            dpg.set_value(f"_color_TF{TFidx+1}", tuple(color_value))
        
    
    def setup_font_theme(self):
        with dpg.font_registry():
            default_font = dpg.add_font("./assets/font/Helvetica.ttf", 16)
        with dpg.theme() as theme_button:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (161, 238, 189))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (174, 255, 204))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (205, 250, 219))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)
        dpg.bind_font(default_font)
        self.theme_button = theme_button

    def get_buffer(self, render_results, mode=None):
        if render_results is None or mode is None:
            output = torch.ones(self.imgH, self.imgW, 3, dtype=torch.float32, device='cuda').detach().cpu().numpy()
        else: 
            output = render_results[mode]
            if mode == "surf_depth":
                output = (output - output.min()) / (output.max() - output.min())
            if len(output.shape) == 2:
                output = output[:,:,None]
            if output.shape[2] == 1:
                output = output.repeat(1, 1, 3)
            if mode in ["diffuse_factor", "specular_factor", "ambient_factor"]:
                opacity = render_results["rend_alpha"].unsqueeze(-1)
                output = opacity*output + (1 - opacity)
            if (self.imgH, self.imgW) != tuple(output.shape[:2]):
                output = self.resize_fn(output)
            output = output.contiguous().detach().cpu().numpy()
        return output

    @property
    def custom_cam(self):
        w2c = self.cam.view
        R = w2c[:3, :3].T
        T = w2c[:3, 3]
        down = self.downsample
        H, W = self.imgH // down, self.imgW // down
        fovx = math.pi * 30 / 180
        fovy = focal2fov(fov2focal(fovx, self.imgH), self.imgW)
        custom_cam = Camera(colmap_id=0, R=R, T=-T,
                            FoVx=fovx, FoVy=fovy, gt_alpha_mask=None,
                            image=torch.zeros(3, H, W), image_name=None, uid=0)
        return custom_cam

    @torch.no_grad()
    def render(self):
        # pause the rendering when segmenting
        if not self.is_segmenting:
            self.step()
        dpg.render_dearpygui_frame()

    def step(self):
        self.start.record()
        render_pkg = self.render_fn(iteration=-1, viewpoint_camera=self.custom_cam, **self.render_kwargs)
        self.end.record()
        torch.cuda.synchronize()
        t = self.start.elapsed_time(self.end)

        buffer1 = self.get_buffer(render_pkg, self.mode)
        self.render_buffer = buffer1

        if t == 0:
            fps = 0
        else:
            fps = int(1000 / t)

        if self.menu is None:
            self.menu_map = {"render": "Blinn-Phong", "rend_normal": "Normal", "surf_depth": "Depth", "rend_alpha": "Alpha",\
                             "diffuse_factor": "diffuse term", "specular_factor": "specular", "ambient_factor": "ambient"}
            self.inv_menu_map = {v: k for k, v in self.menu_map.items()}
            self.menu = ["Blinn-Phong", "Normal", "Depth", "Alpha"]
            
        else:
            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({fps} FPS)')
            dpg.set_value("_texture", self.render_buffer)
    
    def add_oneTFSlider(self, TFidx):
        def callback_TF_slider(sender, app_data):
            TFidx = int(sender.replace("_slider_TF", "")) - 1
            self.focus_TF_idx = TFidx
            with torch.no_grad():
                self.render_kwargs["transform_dict"]["opacity_factors"][TFidx].opacity_factor = torch.tensor(app_data, dtype=torch.float32, device="cuda")
            self.need_update = True
            self._sync_text_label()
        
        def callback_TF_color_edit(sender, app_data):
            TFidx = int(sender.replace("_color_TF", "")) - 1
            self.focus_TF_idx = TFidx
            with torch.no_grad():
                self.render_kwargs["transform_dict"]["palette_colors"][TFidx].palette_color = torch.tensor(app_data[:3], dtype=torch.float32, device="cuda")
            self.need_update = True
            self._sync_text_label()
        
        slider_tag = "_slider_TF" + str(TFidx+1)
        color_tag = "_color_TF" + str(TFidx+1)
        defualt_color = self.render_kwargs["transform_dict"]["palette_colors"][TFidx].palette_color.detach().cpu().numpy()
        defualt_color = (defualt_color * 255).astype(np.uint8).tolist()
        
        indent = 0
        slider_width = 100

        with movable_group(title=f"TF{TFidx+1}", title_color=(0, 0, 0, 255), title_indent=indent+slider_width//4, width=slider_width, height=200, parent="TFs_group"):
            dpg.add_slider_float(
                tag=slider_tag,
                label='',
                default_value=0,
                min_value=0,
                max_value=2.0,
                height=200,
                callback=callback_TF_slider,
                vertical=True,
                width=slider_width, 
                indent=indent
            )
            dpg.add_color_edit(tag=color_tag, default_value=defualt_color, callback=callback_TF_color_edit,
                               no_inputs=True, no_label=True, no_alpha=True, indent=indent+slider_width//4)

    def _sync_text_label(self):
        dpg.set_value("_texture_TF_idx", f"Texture for TF{self.focus_TF_idx+1}")       
        dpg.set_value("_segment_TF_idx", f"Segment TF{self.focus_TF_idx+1}")      

    def register_dpg(self):

        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.imgW, self.imgH, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.imgW, height=self.imgH):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=self.ctrlW, height=self.imgH, pos=(self.imgW, 0),
                        no_resize=True, no_move=True, no_title_bar=True, no_background=True):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            with dpg.group(horizontal=True):
                dpg.add_text("Inference Time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True, leaf=True):
                def callback_change_mode(sender, app_data):
                    self.mode = self.inv_menu_map[app_data]
                    self.need_update = True
                    
                def callback_set_BG_color(sender, app_data):
                    bg_color = app_data[:3]
                    bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                    self.render_kwargs["bg_color"] = bg_color
                    self.need_update = True
                
                def callback_reset_view(sender, app_data):
                    self.cam.reset_view()
                    self.need_update = True
                
                def callback_save_image(sender, app_data):
                    rendered_img = self.render_buffer
                    rendered_img = (rendered_img*255).astype(np.uint8)[...,[2,1,0]]
                    cv2.imwrite(os.path.join("./GUI_results", f'rendered_img.png'), rendered_img)
                    print("Image Saved")
                  
                with dpg.group(horizontal=True):
                    dpg.add_text("Mode")
                    dpg.add_combo(self.menu, indent=self.widget_top, label='', default_value="Blinn-Phong", callback=callback_change_mode)
                    
                with dpg.group(horizontal=True):
                    dpg.add_text("Background Color")
                    dpg.add_color_edit(label="", no_alpha=True, default_value=[255, 255, 255],
                                       indent=self.widget_top, callback=callback_set_BG_color) 
                
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Reset View", tag="_button_reset_view", width=self.ctrlW//2, callback=callback_reset_view)
                    dpg.add_button(label="Save Image", tag="_button_save_image",width=self.ctrlW//2, callback=callback_save_image)
                    dpg.bind_item_theme("_button_reset_view", self.theme_button)
                    dpg.bind_item_theme("_button_save_image", self.theme_button)
            
            
    
            with dpg.collapsing_header(label="Texture & Segmentation", default_open=True, leaf=True):
                def callback_reset_texture(sender, app_data):
                    self.render_kwargs['pc'] = self.original_pc
                    with torch.no_grad():
                            for TFidx in range(self.TFnums):
                                self.render_kwargs["transform_dict"]["opacity_factors"][TFidx].opacity_factor = torch.tensor(1.0, dtype=torch.float32, device="cuda")
                                self.render_kwargs["transform_dict"]["palette_colors"][TFidx].palette_color = self.original_palette_colors[TFidx]
                                color_value = [int(x*255) for x in self.original_palette_colors[TFidx].detach().cpu().numpy()]
                                dpg.set_value(f"_slider_TF{TFidx+1}", 1)
                                dpg.set_value(f"_color_TF{TFidx+1}", tuple(color_value))
                    self.need_update = True
                
                with dpg.group(horizontal=True):
                    dpg.add_text(f"Texture for TF{self.focus_TF_idx+1}", tag="_texture_TF_idx")
                    self._init_file_dialog()
                    dpg.add_button(label="Load", tag="_button_load_texture", indent=self.widget_top, width=self.ctrlW//4, callback=lambda: dpg.show_item("_file_dialog_id"))
                    dpg.add_button(label="Reset", tag="_button_reset_texture",  width=self.ctrlW//4, callback=callback_reset_texture)
                    dpg.bind_item_theme("_button_load_texture", self.theme_button)
                    dpg.bind_item_theme("_button_reset_texture", self.theme_button)

                
    
                def callback_start_rendering_semgentation(sender, app_data):
                    if not self.is_segmenting:
                        def check_valid():
                            if self.mode != "render":
                                return False
                            for TFidx in range(self.TFnums):
                                if TFidx == self.focus_TF_idx:
                                    continue
                                else:
                                    if dpg.get_value(f"_slider_TF{TFidx+1}") != 0:
                                        return False
                            return True
                        if not check_valid():
                            print("Warning: Please set opacity factors of all TFs except the focus TF to zero and mode to Blinn-Phong first.")
                            return
                        self.is_segmenting = True
                        dpg.configure_item(self.start_segment_labelid, label="Finished")
                        print("Log: Start segmenting")
                    else:
                        self.input_label.pop()
                        self.input_points.pop()
                        
                        self.sam_predictor.set_image(self.render_buffer)
                        masks, scores, logits = self.sam_predictor.predict(
                            point_coords = np.array(self.input_points),
                            point_labels = np.array(self.input_label),
                            multimask_output=False
                        )
                        intermediate_render = apply_mask_and_draw_circles(self.render_buffer, masks, self.input_points, self.input_label)
                        dpg.set_value("_texture", intermediate_render)

                        self.render_kwargs['pc'], self.render_kwargs['transform_dict'], self.temp_save_kwargs = split_compose_gaussians(self.render_kwargs['pc'], self.focus_TF_idx, self.render_fn, self.render_kwargs, self.custom_cam, self.ensemble_cameras, self.sam_predictor, self.input_points, self.input_label)
                        self.is_segmenting = False
                        self.input_label = []
                        self.input_points = []
                        
                        self.TFnums += 1
                        self.add_oneTFSlider(self.TFnums-1)
                        self._sync_opacity_color()
                        dpg.configure_item(self.start_segment_labelid, label="Start")
                        print("Log: End segmenting")
                        
                def callback_save_model(sender, app_data):
                    save_segmentations(self.temp_save_kwargs, self.render_kwargs['transform_dict'], "./GUI_results/model")
                    print("Log: Model saved to ./GUI_results/model")
                    
                    
                with dpg.group(horizontal=True):
                    dpg.add_text(f"Segment TF{self.focus_TF_idx+1}", tag="_segment_TF_idx")
                    self.start_segment_labelid=dpg.add_button(label="Start", tag="_rendering_sam", indent=self.widget_top, width=self.ctrlW//2+10, callback=callback_start_rendering_semgentation)
                    dpg.bind_item_theme("_rendering_sam", self.theme_button)
                
                with dpg.group(horizontal=True):
                    dpg.add_text(f"Save Segmentation", tag="_save_model")
                    dpg.add_button(label="Save", tag="_save_model_button", indent=self.widget_top, width=self.ctrlW//2+10, callback=callback_save_model)
                    dpg.bind_item_theme("_save_model_button", self.theme_button)
        
                
                                 
            # color & opacity editing
            with dpg.collapsing_header(label="Color & Opacity Editing", default_open=True, leaf=True):
                with dpg.child_window(width=self.ctrlW-20, height=270, horizontal_scrollbar=True, border=False):
                    with dpg.group(horizontal=True, horizontal_spacing=0, tag='TFs_group'):
                        for i in range(self.TFnums):
                            self.add_oneTFSlider(i)
                    def callback_reset_color_opacity(sender, app_data):
                        with torch.no_grad():
                            for TFidx in range(self.TFnums):
                                self.render_kwargs["transform_dict"]["opacity_factors"][TFidx].opacity_factor = torch.tensor(1.0, dtype=torch.float32, device="cuda")
                                self.render_kwargs["transform_dict"]["palette_colors"][TFidx].palette_color = self.original_palette_colors[TFidx]
                                color_value = [int(x*255) for x in self.original_palette_colors[TFidx].detach().cpu().numpy()]
                                dpg.set_value(f"_slider_TF{TFidx+1}", 1)
                                dpg.set_value(f"_color_TF{TFidx+1}", tuple(color_value))
                        self.need_update = True
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Reset Color & Opacity", tag="_button_reset_color_opacity",width=self.ctrlW-15, callback=callback_reset_color_opacity)
                    dpg.bind_item_theme("_button_reset_color_opacity", self.theme_button)
            # light editing
            with dpg.collapsing_header(label="Light Editing", default_open=True, leaf=True):
                def callback_headlight(sender, app_data):
                    if app_data == False:
                        self.useHeadlight = app_data
                        self.render_kwargs["transform_dict"]["light_transform"].set_light_theta_phi(self.light_angle, self.light_elevation)
                    else:
                        self.useHeadlight = app_data
                    self.render_kwargs["transform_dict"]["light_transform"].useHeadLight = self.useHeadlight
                    self.need_update = True
                with dpg.group(horizontal=True):
                    dpg.add_text("Headlight")
                    dpg.add_checkbox(label="", tag="_checkbox_headlight", callback=callback_headlight, default_value=self.useHeadlight)
                
                def callback_light_angle(sender, app_data):
                    if self.useHeadlight:
                        return
                    if sender == "_slider_light_angle":
                        self.light_angle = app_data
                    else:
                        self.light_elevation = app_data
                    
                    self.render_kwargs["transform_dict"]["light_transform"].set_light_theta_phi(self.light_angle, self.light_elevation)
                    self.need_update = True
                with dpg.group(horizontal=True):
                    dpg.add_text("Azimuthal")
                    dpg.add_slider_int(label="", tag="_slider_light_angle", indent=self.widget_indent,
                                       default_value=self.light_angle, min_value=-180, max_value=180, callback=callback_light_angle)
                with dpg.group(horizontal=True):
                    dpg.add_text("Polar")
                    dpg.add_slider_int(label="", tag="_slider_light_elevation", indent=self.widget_indent,
                                       default_value=self.light_elevation, min_value=-90, max_value=90, callback=callback_light_angle)
                
                def callback_light_multi(sender, app_data):
                    if sender == "_slider_ambient_multi":
                        self.render_kwargs["transform_dict"]["light_transform"].ambient_multi = torch.tensor(app_data, dtype=torch.float32, device="cuda")
                    elif sender == "_slider_light_intensity_multi":
                        self.render_kwargs["transform_dict"]["light_transform"].light_intensity_multi = torch.tensor(app_data, dtype=torch.float32, device="cuda")
                    elif sender == "_slider_specular_multi":
                        self.render_kwargs["transform_dict"]["light_transform"].specular_multi = torch.tensor(app_data, dtype=torch.float32, device="cuda")
                    elif sender == "_slider_shininess_multi":
                        self.render_kwargs["transform_dict"]["light_transform"].shininess_multi = torch.tensor(app_data, dtype=torch.float32, device="cuda")
                    self.need_update = True
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Ambient")
                    dpg.add_slider_float(label="", tag="_slider_ambient_multi", indent=self.widget_indent, default_value=1.0, min_value=0, max_value=5, callback=callback_light_multi)
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Diffuse")
                    dpg.add_slider_float(label="", tag="_slider_light_intensity_multi", indent=self.widget_indent, default_value=1.0, min_value=0, max_value=5, callback=callback_light_multi)
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Specular")
                    dpg.add_slider_float(label="", tag="_slider_specular_multi", indent=self.widget_indent, default_value=1.0, min_value=0, max_value=5, callback=callback_light_multi)
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Shininess")
                    dpg.add_slider_float(label="", tag="_slider_shininess_multi", indent=self.widget_indent, default_value=1.0, min_value=0, max_value=5, callback=callback_light_multi)
            
            # debug info
            if self.debug:
                with dpg.collapsing_header(label="Debug"):
                    # pose
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose), tag="_log_pose")

        ### register camera handler
        def callback_left_mouse_click(sender, app_data):
            if not (dpg.is_item_focused("_primary_window") and self.is_segmenting):
                return
            mouse_x, mouse_y = dpg.get_mouse_pos(local=True)
            mouse_y += 20
            print(f"Left Mouse Clicked at: {mouse_x}, {mouse_y}")
            self.input_points.append([mouse_x, mouse_y])
            self.input_label.append(1)
            self.sam_predictor.set_image(self.render_buffer)
            masks, scores, logits = self.sam_predictor.predict(
                point_coords = np.array(self.input_points),
                point_labels = np.array(self.input_label),
                multimask_output=False
            )
            intermediate_render = apply_mask_and_draw_circles(self.render_buffer, masks, self.input_points, self.input_label)
            dpg.set_value("_texture", intermediate_render)
        
        def callback_right_mouse_click(sender, app_data):
            if not (dpg.is_item_focused("_primary_window") and self.is_segmenting):
                return
            mouse_x, mouse_y = dpg.get_mouse_pos(local=True)
            mouse_y += 20

            print(f"Right Mouse Clicked at: {mouse_x}, {mouse_y}")
            self.input_points.append([mouse_x, mouse_y])
            self.input_label.append(0)
            self.sam_predictor.set_image(self.render_buffer)
            masks, scores, logits = self.sam_predictor.predict(
                point_coords = np.array(self.input_points),
                point_labels = np.array(self.input_label),
                multimask_output=False
            )
            intermediate_render = apply_mask_and_draw_circles(self.render_buffer, masks, self.input_points, self.input_label)
            dpg.set_value("_texture", intermediate_render)
      


        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return
            MouseX, MouseY = dpg.get_mouse_pos()
            x = -(MouseX/ self.imgW - 0.5) * 2
            y = -(MouseY/ self.imgH - 0.5) * 2

            if(self.prev_mouseX is None or self.prev_mouseY is None):
                self.prev_mouseX = x
                self.prev_mouseY = y
                return
            
            self.cam.orbit(self.prev_mouseX, self.prev_mouseY, x, y)
            self.prev_mouseX = x
            self.prev_mouseY = y
            
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))
                
        
        def callback_camera_end_rotate(sender, app_data):
            self.prev_mouseX = None
            self.prev_mouseY = None
        
        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=callback_left_mouse_click)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=callback_right_mouse_click)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)

            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_end_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_camera_drag_pan)

        dpg.create_viewport(title='TexGS-VolVis', width=self.imgW+self.ctrlW, height=self.imgH, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()
        dpg.show_viewport()


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument('-so', '--source_dir', default=None, required=True, help="the source ckpts dir")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('-t', '--type', choices=['2DGS', 'TexGS', 'stylize', 'stylize_inf'], default='TexGS')
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--gui_debug", action="store_true", help="show debug info in GUI")
    args = parser.parse_args()
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    
    DEFAULT_FD_DIR = "/home/dullpigeon/Desktop/Spring/InstructGS2GS-Texture2DGS/output/textures"
    
    pbr_kwargs = dict()
    scene_dict = load_ckpts_paths(args.source_dir)
    TFs_names = list(scene_dict.keys())
    TFs_nums = len(TFs_names)
    palette_color_transforms = []
    opacity_transforms = []
    TFcount=0
    for TFs_name in TFs_names:
        palette_color_transform = LearningPaletteColor()
        palette_color_transform.create_from_ckpt(f"{scene_dict[TFs_name]['palette']}")
        palette_color_transforms.append(palette_color_transform)
        opacity_factor=0.0 if TFcount not in [] else 1.0
        opacity_transform = LearningOpacityTransform(opacity_factor=opacity_factor)
        opacity_transforms.append(opacity_transform)
        TFcount+=1
    if args.type == "stylize":
        light_transform = LearningLightTransform(theta=180, phi=0, ambient_multi=1.0, light_intensity_multi=1.0, specular_multi=1.0)
    else:
        light_transform = LearningLightTransform(theta=180, phi=0)
    # load gaussians
    gaussians_composite, gs_list_kwargs = texGS_scene_composition(scene_dict, dataset)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    render_kwargs = {
        "pc": gaussians_composite,
        "pipe": pipe,
        "bg_color": background,
        "is_training": False,
        "transform_dict": {
            "palette_colors": palette_color_transforms,
            "opacity_factors": opacity_transforms,
            "light_transform": light_transform
        }
    }
    dataset_kwargs = {
        "ensemble_json_views_path": "/home/dullpigeon/Desktop/Spring/InstructGS2GS-Texture2DGS/assets/icosphere_views"
    }
        
    render_fn = render_fn_dict[args.type]
    

    windows = GUI(args, render_fn=render_fn, render_kwargs=render_kwargs, dataset_kwargs=dataset_kwargs, TFnums=TFs_nums, gs_list_kwargs=gs_list_kwargs)
    
    while dpg.is_dearpygui_running():
        windows.render()