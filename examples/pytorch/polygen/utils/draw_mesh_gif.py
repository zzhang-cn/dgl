import os
import sys
import numpy as np
import imageio
from PIL import Image

import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.io.obj_io import load_obj as load_3d_obj

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer.cameras import OpenGLPerspectiveCameras, look_at_view_transform
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer.mesh.renderer import MeshRenderer
from pytorch3d.renderer.mesh.shader import (
    BlendParams,
    HardFlatShader,
    HardGouraudShader,
    HardPhongShader,
    SoftSilhouetteShader,
    TexturedSoftPhongShader,
)
from pytorch3d.renderer.mesh.texturing import Textures
from pytorch3d.structures.meshes import Meshes
from pytorch3d.utils.ico_sphere import ico_sphere

import matplotlib.pyplot as plt

def draw_mesh_gif(obj_file_path, num_frames=20):
    R, T = look_at_view_transform(dist=2.7, elev=45.0, azim=45.0)

    device = torch.device(type='cpu')
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

    # Init shader settings
    materials = Materials(device=device)
    lights = PointLights(device=device)
    lights.location = torch.tensor([0.0, 0.0, +2.0], device=device)[None]
    shader = HardGouraudShader(lights=lights, cameras=cameras, materials=materials)

    raster_settings = RasterizationSettings(
        image_size=512, blur_radius=0.0, faces_per_pixel=1, bin_size=0
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    # load mesh
    verts, faces, aux = load_3d_obj(obj_file_path)

    num_faces_per_frame = faces[0].shape[0] // num_frames
    imgs = []
    for i in range(num_frames):
        faces_to_show = (i+1)*num_faces_per_frame if i < num_frames - 1 else faces[0].shape[0]
        pred_mesh = Meshes(
                verts=[verts],
                faces=[faces[0][:faces_to_show]],
                )
        verts_padded = pred_mesh.verts_padded()
        faces_padded = pred_mesh.faces_padded()
        textures = Textures(verts_rgb=torch.ones_like(verts_padded))
        pred_mesh = Meshes(verts=verts_padded, faces=faces_padded, textures=textures)
        image = renderer(pred_mesh).data.numpy()
        np_image = (image[0,::-1,:,:3]*255).astype(np.uint8)
        im = Image.fromarray(np.transpose(np_image, (1, 0, 2))[::-1,:,:])
        imgs.append(np.transpose(np_image, (1, 0, 2))[::-1,:,:])
    for i in range(10):
        imgs.append(np.transpose(np_image, (1, 0, 2))[::-1,:,:])


    gif_file_path = obj_file_path[:-3] + 'gif'
    #imgs[0].save(gif_file_path, save_all=True, append_images=imgs)
    imageio.mimsave(gif_file_path, imgs)

if __name__ == '__main__':
    draw_mesh_gif(sys.argv[1])
