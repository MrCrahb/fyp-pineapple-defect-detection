import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageChops
import noise  # For Perlin noise
# from scipy.spatial import KDTree  # No longer needed for base
import os
import io
import zipfile
import random
import time

# --- Constants ---
# Colors based on your report's "Ripeness Spectrum"
# We now have a "BASE" color and a "SCALE" (eye) color
COLOR_UNRIPE_BASE = np.array([40, 80, 20])
COLOR_UNRIPE_SCALE = np.array([30, 60, 15]) # Darker, duller green

COLOR_RIPE_BASE = np.array([220, 180, 20])
COLOR_RIPE_SCALE = np.array([160, 110, 30]) # Brownish-yellow scale

COLOR_OVERRIPE_BASE = np.array([130, 90, 30])
COLOR_OVERRIPE_SCALE = np.array([80, 60, 20]) # Dark, dull brown

# --- Defect Colors (from your real-world example) ---
COLOR_WHITE_FUNGUS = np.array([210, 210, 210])
COLOR_BLACK_SPOTS = np.array([20, 10, 5])
# COLOR_HOLE = np.array([15, 10, 5])   # Removed as requested

COLOR_BASE_STUMP = np.array([50, 30, 10]) # Renamed from COLOR_BASE

# --- Procedural Generation Functions ---

def generate_tiled_scale_mask(width, height, scale_size):
    """
    Generates the base pineapple "eye" texture using a repeating
    tiled (hexagonal/diamond) grid.
    This replaces the Voronoi function for a realistic, structured pattern.
    """
    
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    
    # Warp coordinates to simulate 3D curvature
    center_x = width / 2
    # Horizontal bulge
    x_offset = (xx - center_x) / (width / 2.0) # Range [-1, 1]
    # Apply a sine wave warp based on vertical position and a horizontal "bulge"
    # This creates the spiral
    xx_warped = xx + np.sin(yy / (height / 6.0)) * 10 * (1 - np.abs(x_offset))
    yy_warped = yy + (xx_warped - center_x) * 0.2 # Add a diagonal slant
    
    # --- New Tiling Logic ---
    # Create a repeating diagonal grid
    # We use sin waves to create soft, repeating blobs
    s = scale_size
    diag1 = np.sin((xx_warped + yy_warped) * (2 * np.pi / s))
    diag2 = np.sin((xx_warped - yy_warped) * (2 * np.pi / s))
    
    # Combine the two diagonal patterns
    # Multiplying them creates a "checkerboard" of peaks and valleys
    grid = diag1 * diag2
    
    # Normalize and threshold to get the "scale" mask
    # We take the "peaks" of the grid pattern
    grid_norm = (grid - grid.min()) / (grid.max() - grid.min())
    
    # Threshold to create the "eye"
    # This creates a mask where > 0.7 is the "scale"
    scale_mask = (grid_norm > 0.7).astype(np.uint8) * 255
    
    return scale_mask


# --- Removed generate_voronoi_base_mask ---

def generate_perlin_white_fungus(width, height, scale, octaves, persistence, lacunarity):
    """
    Generates Perlin noise for "white fungus".
    """
    mold_img = np.zeros((height, width))
    seed = np.random.randint(0, 100)
    for y in range(height):
        for x in range(width):
            mold_img[y][x] = noise.pnoise2(
                x / scale,
                y / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=seed
            )
            
    mold_img = ((mold_img + 1) / 2 * 255) # Normalize from [-1, 1] to [0, 255]
    
    # --- FIX: Use dynamic percentile threshold ---
    # This guarantees a defect is always generated.
    threshold = np.percentile(mold_img, 95) # Top 5% of noise becomes fungus
    mold_img = np.where(mold_img > threshold, 255, 0)
    
    return Image.fromarray(mold_img.astype(np.uint8))

def generate_perlin_black_spots(width, height, scale, octaves, persistence, lacunarity):
    """
    Generates Perlin noise for "black spots".
    Uses different parameters for a different look.
    """
    spot_img = np.zeros((height, width))
    seed = np.random.randint(100, 200) # Different seed
    for y in range(height):
        for x in range(width):
            spot_img[y][x] = noise.pnoise2(
                x / scale,
                y / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=seed
            )
            
    spot_img = ((spot_img + 1) / 2 * 255) # Normalize from [-1, 1] to [0, 255]
    
    # --- FIX: Use dynamic percentile threshold ---
    # We use a higher percentile to make the spots smaller and rarer.
    threshold = np.percentile(spot_img, 98) # Top 2% of noise becomes spots
    spot_img = np.where(spot_img > threshold, 255, 0)
    
    return Image.fromarray(spot_img.astype(np.uint8))


def apply_ripeness_color(base_mask, ripeness_factor):
    """
    Applies the green -> yellow -> brown gradient.
    This is the NEW logic. It creates two colors (base, scale) and combines them.
    """
    
    # 1. Interpolate the BASE color
    if ripeness_factor < 0.5:
        factor = ripeness_factor * 2
        base_color = (1 - factor) * COLOR_UNRIPE_BASE + factor * COLOR_RIPE_BASE
    else:
        factor = (ripeness_factor - 0.5) * 2
        base_color = (1 - factor) * COLOR_RIPE_BASE + factor * COLOR_OVERRIPE_BASE
        
    # 2. Interpolate the SCALE color
    if ripeness_factor < 0.5:
        factor = ripeness_factor * 2
        scale_color = (1 - factor) * COLOR_UNRIPE_SCALE + factor * COLOR_RIPE_SCALE
    else:
        factor = (ripeness_factor - 0.5) * 2
        scale_color = (1 - factor) * COLOR_RIPE_SCALE + factor * COLOR_OVERRIPE_SCALE

    # 3. Create the image by combining colors
    # Create a 3-channel image, default to base color
    color_img = np.zeros((base_mask.shape[0], base_mask.shape[1], 3), dtype=np.uint8)
    color_img[:, :] = base_color.astype(np.uint8)
    
    # Use the mask to apply the scale color
    color_img[base_mask == 255] = scale_color.astype(np.uint8)

    return color_img.astype(np.uint8)

def create_3d_shading_and_mask(width, height):
    """
    Creates a "2.5D" shading map and an oval mask
    to simulate a 3D pineapple body.
    """
    # Create X, Y grids from -1 to 1
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)
    
    # Create an oval mask
    mask = (xx**2 + (yy / 0.9)**2) < 0.8
    
    # Create shading (brighter in the middle, darker at edges)
    shading = (np.cos(xx * np.pi / 2) + 1) / 2
    # Add a slight vertical highlight
    shading = shading * ((np.cos(yy * np.pi / 2.2) + 1) / 2 * 0.5 + 0.5)
    shading = (shading * 255).astype(np.uint8)
    
    # --- FIX: Cast mask to np.uint8 to ensure 'L' mode ---
    # This prevents the "illegal image mode" error in putalpha
    return Image.fromarray(shading), Image.fromarray((mask * 255).astype(np.uint8))

def generate_base(width, height):
    """
    Generates a procedural "base" (bottom stump).
    """
    base_height = int(height * 0.1)
    base_canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(base_canvas)
    
    base_y = int(height * 0.9)
    center_x = width / 2
    base_width = width / 8
    draw.ellipse(
        (center_x - base_width, base_y - base_height/2, 
         center_x + base_width, base_y + base_height/2),
        fill=tuple(COLOR_BASE_STUMP)
    )
    return base_canvas


def generate_synthetic_pineapple(width, height, params, is_preview=False):
    """
    Main orchestration function.
    Generates a single pineapple image for classification.
    
    Args:
        width (int): Image width.
        height (int): Image height.
        params (dict): Dictionary of generation parameters.
        is_preview (bool): If True, disables randomization for predictable
                           previews (e.g., defect placement).
    """
    
    # 0. Create a final RGBA canvas
    final_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    
    # --- COMPOSITING FIX: Paste base *first* ---
    # 1. Generate and add base (goes under the body)
    base_img = generate_base(width, height)
    final_image.paste(base_img, (0, 0), base_img)

    # 2. Generate 3D shading and body mask
    shading, body_mask = create_3d_shading_and_mask(width, height)
    
    # 3. Generate Base Texture MASK (using new function)
    base_texture_mask = generate_tiled_scale_mask(width, height, params['scale_size'])
    
    # 4. Apply Ripeness Color (New function)
    pineapple_img_color = apply_ripeness_color(base_texture_mask, params['ripeness'])
    pineapple_img = Image.fromarray(pineapple_img_color)
    
    # 5. Apply 3D Shading
    shading_l = shading.convert('L')
    pineapple_img = ImageChops.multiply(pineapple_img, shading_l.convert('RGB'))
    
    # 6. Apply Defects (before masking)
    if params['has_white_fungus']:
        mold_mask = generate_perlin_white_fungus(
            width, height, 
            params['fungus_scale'], params['fungus_octaves'], 
            params['fungus_persistence'], params['fungus_lacunarity']
        )
        # Create the colored layer
        mold_layer = Image.new("RGB", (width, height), tuple(COLOR_WHITE_FUNGUS))
        
        # --- FIX: Use predictable placement for previews ---
        if is_preview:
            paste_x, paste_y = 0, 0 # Center it for preview
        else:
            paste_x = random.randint(-width // 4, width // 4) # Randomize for batch
            paste_y = random.randint(-height // 4, height // 4)
            
        pineapple_img.paste(mold_layer, (paste_x, paste_y), mold_mask) # Use mask here

    if params['has_black_spot']:
        spot_mask = generate_perlin_black_spots(
            width, height, 
            params['spot_scale'], params['spot_octaves'], 
            params['spot_persistence'], params['spot_lacunarity']
        )
        # Create the colored layer
        spot_layer = Image.new("RGB", (width, height), tuple(COLOR_BLACK_SPOTS))

        # --- FIX: Use predictable placement for previews ---
        if is_preview:
            paste_x, paste_y = 0, 0 # Center it for preview
        else:
            paste_x = random.randint(-width // 4, width // 4) # Randomize for batch
            paste_y = random.randint(-height // 4, height // 4)
            
        pineapple_img.paste(spot_layer, (paste_x, paste_y), spot_mask) # Use mask here

    # --- "has_hole" section removed ---
    
    # 7. Apply body mask to the pineapple body
    pineapple_img.putalpha(body_mask)
    final_image.paste(pineapple_img, (0, 0), pineapple_img) # Paste pineapple *over* base

    # 8. Generate and add crown (goes on top)
    # --- CROWN REMOVED AS REQUESTED ---
    # crown_img = generate_crown(width, height)
    # final_image.paste(crown_img, (0, 0), crown_img)
    
    # Convert to RGB with a white background for non-transparent saving
    rgb_image = Image.new("RGB", final_image.size, (255, 255, 255))
    rgb_image.paste(final_image, mask=final_image.split()[3]) # Paste using alpha
        
    return rgb_image

# --- Streamlit GUI ---

st.set_page_config(layout="wide")
st.title("🍍 Pineapple Synthetic Data Generator")
st.write("This tool generates synthetic pineapple images for **classification** with multiple defect types.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Live Preview & Parameter Tuning")
    st.write("Use these controls to test parameters as per your 'Data Understanding' phase.")

    # Sliders for live preview
    preview_ripeness = st.slider("Ripeness (0=Green, 0.5=Yellow, 1=Brown)", 0.0, 1.0, 0.5, 0.05)
    
    preview_has_fungus = st.checkbox("Show White Fungus", value=True)
    preview_has_spot = st.checkbox("Show Black Spots")
    # --- "Show Holes" checkbox removed ---

    st.subheader("Advanced Noise Parameters")
    with st.expander("Tune Noise Parameters"):
        p_img_width = st.number_input("Image Width", 128, 1024, 300)
        p_img_height = st.number_input("Image Height", 128, 1024, 450)
        
        st.markdown("---")
        st.write("**Base Texture (Tiled 'Eyes')**")
        # --- Replaced 'base_points' with 'scale_size' ---
        p_scale_size = st.slider("Scale Size", 10.0, 100.0,80.0, 1.0)
        
        st.markdown("---")
        st.write("**White Fungus (Perlin Noise)**")
        p_fungus_scale = st.slider("Fungus Scale", 20.0, 200.0, 60.0)
        p_fungus_octaves = st.slider("Fungus Octaves", 1, 10, 6)
        p_fungus_persistence = st.slider("Fungus Persistence", 0.1, 1.0, 0.5)
        p_fungus_lacunarity = st.slider("Fungus Lacunarity", 1.0, 5.0, 2.0)
        
        st.markdown("---")
        st.write("**Black Spots (Perlin Noise)**")
        p_spot_scale = st.slider("Spot Scale", 10.0, 100.0, 65.0)
        p_spot_octaves = st.slider("Spot Octaves", 1, 10, 6)
        p_spot_persistence = st.slider("Spot Persistence", 0.1, 1.0, 0.5)
        p_spot_lacunarity = st.slider("Spot Lacunarity", 1.0, 5.0, 1.5)
        
        # --- "Holes" sliders removed ---

    if st.button("Generate Preview", key="preview"):
        params = {
            'ripeness': preview_ripeness,
            'has_white_fungus': preview_has_fungus,
            'has_black_spot': preview_has_spot,
            'scale_size': p_scale_size, # <-- New parameter
            'fungus_scale': p_fungus_scale,
            'fungus_octaves': p_fungus_octaves,
            'fungus_persistence': p_fungus_persistence,
            'fungus_lacunarity': p_fungus_lacunarity,
            'spot_scale': p_spot_scale,
            'spot_octaves': p_spot_octaves,
            'spot_persistence': p_spot_persistence,
            'spot_lacunarity': p_spot_lacunarity,
        }
        
        with st.spinner("Generating preview..."):
            # --- FIX: Call with is_preview=True ---
            img = generate_synthetic_pineapple(p_img_width, p_img_height, params, is_preview=True)
            st.image(img, caption="Generated Preview", use_column_width=True)


with col2:
    st.header("Batch Generation for Dataset")
    st.write("Select the quantities for each ripeness state. Defects will be added based on the probabilities you set.")

    img_width = st.number_input("Image Width", 128, 1024, 300, key="b_w")
    img_height = st.number_input("Image Height", 128, 1024, 450, key="b_h")
    
    st.markdown("---")
    st.subheader("Categories")
    
    categories = {
        "unripe": st.number_input("Images: Unripe", 0, 1000, 0),
        "ripe": st.number_input("Images: Ripe", 0, 1000, 0),
        "overripe": st.number_input("Images: Overripe", 0, 1000, 0),
    }
    
    st.markdown("---")
    st.subheader("Defect Probabilities")
    st.write("For each image generated, what is the chance it has a defect?")
    prob_fungus = st.slider("Chance of White Fungus", 0.0, 1.0, 0.3, 0.05)
    prob_spot = st.slider("Chance of Black Spots", 0.0, 1.0, 0.2, 0.05)
    # --- "Chance of Holes" slider removed ---
    
    total_images = sum(categories.values())
    st.info(f"Total images to be generated: **{total_images}**")

    if st.button("Generate Dataset", key="batch") and total_images > 0:
        
        # Use default (tuned) noise parameters from the expander
        default_params = {
            'scale_size': p_scale_size, # <-- New parameter
            'fungus_scale': p_fungus_scale,
            'fungus_octaves': p_fungus_octaves,
            'fungus_persistence': p_fungus_persistence,
            'fungus_lacunarity': p_fungus_lacunarity,
            'spot_scale': p_spot_scale,
            'spot_octaves': p_spot_octaves,
            'spot_persistence': p_spot_persistence,
            'spot_lacunarity': p_spot_lacunarity,
        }
        
        zip_buffer = io.BytesIO()
        progress_bar = st.progress(0.0)
        progress_text = st.empty()
        generated_count = 0

        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for category, count in categories.items():
                if count == 0:
                    continue
                
                for i in range(count):
                    params = default_params.copy()
                    
                    # Set ripeness
                    if "unripe" in category:
                        params['ripeness'] = random.uniform(0.0, 0.3)
                    elif "ripe" in category:
                        params['ripeness'] = random.uniform(0.4, 0.6)
                    elif "overripe" in category:
                        params['ripeness'] = random.uniform(0.7, 1.0)
                    
                    # Set defects based on probability
                    params['has_white_fungus'] = random.random() < prob_fungus
                    params['has_black_spot'] = random.random() < prob_spot
                    # params['has_hole'] = random.random() < prob_hole # Removed
                    
                    # --- Build filename based on defects ---
                    defect_names = []
                    if params['has_white_fungus']:
                        defect_names.append("white_fungus")
                    if params['has_black_spot']:
                        defect_names.append("black_spot")
                    # if params['has_hole']: # Removed
                    #     defect_names.append("hole") # Removed
                    
                    if not defect_names:
                        defect_str = "healthy"
                    else:
                        defect_str = "_".join(defect_names)
                        
                    final_category = f"{category}_{defect_str}"
                    # --- End of filename logic ---
                    
                    # Domain randomization for noise parameters
                    params['scale_size'] = random.uniform(0.9, 1.1) * default_params['scale_size'] # <-- New parameter
                    params['fungus_scale'] = random.uniform(0.8, 1.2) * default_params['fungus_scale']
                    params['spot_scale'] = random.uniform(0.8, 1.2) * default_params['spot_scale']

                    # --- FIX: Batch generation calls with is_preview=False (default) ---
                    img = generate_synthetic_pineapple(img_width, img_height, params) # is_preview is False by default
                    
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, "PNG")
                    
                    base_filename = f"{final_category}/{final_category}_{i+1:04d}"
                    zip_file.writestr(f"{base_filename}.png", img_buffer.getvalue())
                    
                    generated_count += 1
                    progress = generated_count / total_images
                    progress_bar.progress(progress)
                    progress_text.text(f"Generated {generated_count} / {total_images} images... ({progress:.1%})")

        progress_text.success(f"Successfully generated {total_images} images!")
        st.balloons()
        
        st.download_button(
            label="Download Dataset (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="synthetic_pineapple_dataset.zip",
            mime="application/zip"
        )

