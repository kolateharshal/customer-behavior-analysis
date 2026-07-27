import fitz
import numpy as np

filepath = "/Users/harshal/Downloads/VocaSense-Voice-Interaction-and-Coding-Assistant.pdf.pdf"
doc = fitz.open(filepath)

for i in range(len(doc)):
    page = doc[i]
    pix = page.get_pixmap(dpi=150)
    
    # Convert pixmap to numpy array
    # pix.samples is a bytes object, we can reconstruct it
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
    
    # We only care about the bottom-right region of the page
    # Let's look at the bottom-right quadrant: x > 0.7 * width, y > 0.8 * height
    x_start = int(0.7 * pix.width)
    y_start = int(0.85 * pix.height)
    
    sub_img = img[y_start:, x_start:]
    
    # Find white or near-white pixels
    # e.g., R, G, B all > 240
    # if it's RGBA, n=4, otherwise n=3
    mask = (sub_img[:, :, 0] > 240) & (sub_img[:, :, 1] > 240) & (sub_img[:, :, 2] > 240)
    
    y_indices, x_indices = np.where(mask)
    if len(x_indices) > 0:
        # Get bounding box in local sub_img coordinates
        min_x, max_x = x_indices.min(), x_indices.max()
        min_y, max_y = y_indices.min(), y_indices.max()
        
        # Convert to page coordinates
        # Map back to full pixmap coordinates
        full_min_x = x_start + min_x
        full_max_x = x_start + max_x
        full_min_y = y_start + min_y
        full_max_y = y_start + max_y
        
        # Convert pixmap coordinates back to PDF points
        # PDF point = pixmap pixel / (dpi / 72.0)
        scale = 150 / 72.0
        pdf_min_x = full_min_x / scale
        pdf_max_x = full_max_x / scale
        pdf_min_y = full_min_y / scale
        pdf_max_y = full_max_y / scale
        
        print(f"Page {i+1}: Badge found at PDF rect: ({pdf_min_x:.2f}, {pdf_min_y:.2f}, {pdf_max_x:.2f}, {pdf_max_y:.2f})")
    else:
        print(f"Page {i+1}: No badge found in bottom-right region.")
