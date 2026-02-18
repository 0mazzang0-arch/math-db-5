from pathlib import Path
import fitz  # PyMuPDF

pdf_path = Path(r"D:\math-db-5\test2.pdf")
out_dir = Path(r"D:\math-db-5\pdf_cutter_output\_pages_all")
out_dir.mkdir(parents=True, exist_ok=True)

doc = fitz.open(pdf_path)
zoom = 2.0
mat = fitz.Matrix(zoom, zoom)

for i in range(len(doc)):
    page = doc.load_page(i)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save((out_dir / f"P{i+1:03d}.png").as_posix())

print("pdf=", pdf_path)
print("out_dir=", out_dir)
print("pages=", len(doc))
