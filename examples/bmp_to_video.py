import cv2
import os
import argparse
import glob
import re

def natural_sort_key(s):
    """
    Sorts strings with embedded numbers naturally (e.g., frame_2.bmp comes before frame_10.bmp).
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def images_to_video(input_folder, output_file, fps=30):
    # 1. Find all BMP files
    search_pattern = os.path.join(input_folder, "*.bmp")
    images = glob.glob(search_pattern)
    
    if not images:
        print(f"Error: No BMP images found in {input_folder}")
        return

    # 2. Sort images naturally
    images.sort(key=natural_sort_key)
    print(f"Found {len(images)} images. Processing...")

    # 3. Read the first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    size = (width, height)
    
    # 4. Initialize VideoWriter
    # mp4v is a safe codec for MP4. For AVI you can use 'MJPG' or 'XVID'.
    if output_file.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    else:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    out = cv2.VideoWriter(output_file, fourcc, fps, size)
    
    count = 0
    for file_path in images:
        img = cv2.imread(file_path)
        if img is None:
            print(f"Warning: Could not read {file_path}, skipping.")
            continue
            
        # Ensure verification of size consistency
        if (img.shape[1], img.shape[0]) != size:
             print(f"Warning: Size mismatch for {file_path}, resizing to match first frame.")
             img = cv2.resize(img, size)
             
        out.write(img)
        count += 1
        if count % 50 == 0:
            print(f"Processed {count} frames...")

    out.release()
    print(f"\nSuccess! Video saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BMP images directory to a Video file.")
    parser.add_argument("-i", "--input", required=True, help="Input directory containing BMP files")
    parser.add_argument("-o", "--output", default="output.mp4", help="Output video filename (default: output.mp4)")
    parser.add_argument("-f", "--fps", type=int, default=30, help="Frames Per Second (default: 30)")
    
    args = parser.parse_args()
    
    images_to_video(args.input, args.output, args.fps)
