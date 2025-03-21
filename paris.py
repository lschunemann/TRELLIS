import os
os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers'>
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', defaul>
                                            # 'auto' is faster but will do benc>
                                            # Recommended to set to 'native' if>
import time
import glob
from pathlib import Path
import torch

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

def process_object(object_name, base_path, output_base_path):
  # Load a pipeline from a model folder or a Hugging Face model hub.
  pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
  pipeline.cuda()

                                                      
  # Process both start and end views
  for view in ['start', 'end']:
      # Get the first 4 images from train folder
      image_path = os.path.join(base_path, object_name, '*', view, 'train', '*.png')
      image_files = sorted(glob.glob(image_path))[:4]
      
      if not image_files:
          print(f"No images found for {object_name} {view}")
          continue
          
      # Create output directory
      output_dir = os.path.join(output_base_path, object_name, view)
      os.makedirs(output_dir, exist_ok=True)
      
      # Process each set of 4 images as different views
      if len(image_files) >= 4:
          print(f"Processing {object_name} {view} images as multi-view set")
          
          # Load all 4 images
          images = [
              Image.open(image_files[0]).convert("RGBA"),
              Image.open(image_files[1]).convert("RGBA"),
              Image.open(image_files[2]).convert("RGBA"),
              Image.open(image_files[3]).convert("RGBA")  # Using 4th image as additional view
          ]
          
          # Generate mesh for the set of images
          start_time = time.time()
          # Run the pipeline
          outputs = pipeline.run_multi_image(
            images,
              seed=1,
              # Optional parameters
              # sparse_structure_sampler_params={
              #     "steps": 12,
              #     "cfg_strength": 7.5,
              # },
              # slat_sampler_params={
              #     "steps": 12,
              #     "cfg_strength": 3,
              # },
          )
          # outputs is a dictionary containing generated 3D assets in different formats:
          # - outputs['gaussian']: a list of 3D Gaussians
          # - outputs['radiance_field']: a list of radiance fields
          # - outputs['mesh']: a list of meshes
          
          # Render the outputs
          video = render_utils.render_video(outputs['gaussian'][0])['color']
          imageio.mimsave(os.path.join(output_dir, f"{object_name}_{view}_gs.mp4"), video, fps=30)
          video = render_utils.render_video(outputs['radiance_field'][0])['color']
          imageio.mimsave(os.path.join(output_dir, f"{object_name}_{view}_rf.mp4"), video, fps=30)
          video = render_utils.render_video(outputs['mesh'][0])['normal']
          imageio.mimsave(os.path.join(output_dir, f"{object_name}_{view}_mesh.mp4"), video, fps=30)
          
          # GLB files can be extracted from the outputs
          glb = postprocessing_utils.to_glb(
              outputs['gaussian'][0],
              outputs['mesh'][0],
              # Optional parameters
              simplify=0.95,          # Ratio of triangles to remove in the simplificatio>
              texture_size=1024,      # Size of the texture used for the GLB
          )
          glb.export(os.path.join(output_dir, f'{object_name}_{view}.glb'))
          
          # Save Gaussians as PLY files
          outputs['gaussian'][0].save_ply(os.path.join(output_dir, f'{object_name}_{view}.ply'))

def main():
    # Base paths
    base_path = '/home/lschuenemann/TRELLIS/data/load/sapien'
    output_base_path = '/home/lschuenemann/TRELLIS/data/data/meshes'
    
    # Get all object directories
    object_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # Process each object
    for object_name in object_dirs:
        print(f"\nProcessing {object_name}...")
        process_object(object_name, base_path, output_base_path)

if __name__ == "__main__":
    main()
