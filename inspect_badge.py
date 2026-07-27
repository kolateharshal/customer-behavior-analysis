import fitz

filepath = "/Users/harshal/Downloads/VocaSense-Voice-Interaction-and-Coding-Assistant.pdf.pdf"
doc = fitz.open(filepath)

for i in range(len(doc)):
    page = doc[i]
    print(f"\n--- Page {i+1} ---")
    print("Annotations:", len(page.annots()))
    print("Links:", len(page.get_links()))
    
    # List all XObjects
    xobjects = page.get_images()
    print("XObjects (images/forms):", len(xobjects))
    for x in xobjects:
        print("  XObject:", x)
