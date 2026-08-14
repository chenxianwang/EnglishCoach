"""Render a retrospective to one PNG per slide.

    python3 retro_slides.py                 # the newest retrospective
    python3 retro_slides.py 2026-08-13      # a particular one, by filename stem
    python3 retro_slides.py 2026-08-13 16   # re-render just slide 16

The retrospectives are written as one continuous page, so "one slide" is a
decision rather than something the file states. Three rules make it:

  * every <article class="slide"> is a slide;
  * the masthead is two slides -- the cover, and the contents page it carries;
  * a section divider gets no image of its own. It is a thin band, worth
    nothing alone, so it rides on top of the slide that opens its section and
    every exported image ends up saying which section it belongs to.

Chrome's --screenshot captures the viewport, not the page, so each slide is
rendered into a deliberately over-tall window and then trimmed back to its own
content. Trimming rather than measuring first is what keeps this one pass: the
paper is a flat colour, so the bounding box of the ink is unambiguous.

Images go to <stem>-slides/ beside the document, named with the deck's own
slide numbers, so the folder reads in presentation order.
"""
import html
import os
import re
import subprocess
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RETRO = os.path.join(HERE, "retrospectives")
CSS = os.path.join(RETRO, "_slides.css")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

WIDTH = 1200          # CSS px; the document's column is 1080 plus its gutters
TALL = 5200           # taller than any slide, so nothing is cut before trimming
SCALE = 2             # retina -- the PNGs come out at twice these numbers
PAD = 44              # CSS px of paper left around the trimmed content
TOL = 6               # how far off the paper colour a pixel counts as ink,
                      # loose enough that the cards' soft shadow does not

PAGE = """<!doctype html>
<html data-theme="light"><head><meta charset="utf-8">
<style>%s
/* export-only: the column's bottom padding exists for scrolling, not stills */
.wrap{padding-bottom:0}
.sec{margin-top:0}
.slide{margin-top:28px}
.mast{padding-top:8px;border-bottom:none}
</style></head><body><div class="wrap">%s</div></body></html>
"""


def slug(text):
    t = html.unescape(re.sub(r"<[^>]+>", " ", text)).lower()
    t = t.replace("’", "").replace("'", "")
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-")[:52]


def slides(doc):
    """The document, cut into (number, slug, markup) in presentation order."""
    mast = re.search(r'<div class="mast">(.*?)\n</div>', doc, re.S).group(1)
    toc = re.search(r'<nav class="toc">.*?</nav>', mast, re.S).group(0)
    out = [("01", "cover", '<div class="mast">%s</div>' % mast.replace(toc, "")),
           ("02", "contents", '<div class="mast">%s</div>' % toc)]

    pending = ""
    for m in re.finditer(r'<div class="sec".*?</div>'
                         r'|<article class="slide"[^>]*>.*?\n</article>', doc, re.S):
        block = m.group(0)
        if block.startswith('<div class="sec"'):
            pending = block + "\n"
            continue
        num = re.search(r'<span class="sn">(.*?)</span>', block).group(1)
        head = re.search(r'<span class="sn">.*?</span><h3>(.*?)</h3>',
                         block, re.S).group(1)
        out.append((num.lower(), slug(head), pending + block))
        pending = ""

    # a slide dropped by one of those patterns is exactly the failure that
    # would go unnoticed until the images were already in use
    assert len(out) == 2 + doc.count('<article class="slide"'), len(out)
    return out


def shoot(page_html, work, stem):
    """Render one page and trim it to its content. Returns the trimmed image."""
    page, shot = os.path.join(work, stem + ".html"), os.path.join(work, stem + ".png")
    with open(page, "w", encoding="utf-8") as f:
        f.write(page_html)
    r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=%d" % SCALE,
                        "--screenshot=" + shot,
                        "--window-size=%d,%d" % (WIDTH, TALL),
                        "file://" + page], capture_output=True, text=True)
    if not os.path.exists(shot):
        sys.exit("chrome failed on %s\n%s" % (stem, r.stderr[-2000:]))

    im = Image.open(shot).convert("RGB")
    paper = im.getpixel((4, 4))
    mask = Image.new("L", im.size, 0)
    mask.putdata([0 if max(abs(p[0] - paper[0]), abs(p[1] - paper[1]),
                           abs(p[2] - paper[2])) < TOL else 255 for p in im.getdata()])
    x0, y0, x1, y1 = mask.getbbox()
    pad = PAD * SCALE
    y0, y1 = max(0, y0 - pad), min(im.height, y1 + pad)
    # full width is kept deliberately: cropping to the ink would re-centre every
    # slide on its own widest element, and the column would jitter between images
    out = Image.new("RGB", (im.width, y1 - y0), paper)
    out.paste(im.crop((0, y0, im.width, y1)), (0, 0))
    return out


def main(argv):
    want_doc = argv[0] if argv else ""
    only = argv[1] if len(argv) > 1 else ""

    docs = sorted((n for n in os.listdir(RETRO)
                   if n.endswith(".html") and want_doc in n), reverse=True)
    if not docs:
        sys.exit("no retrospective matching %r in %s" % (want_doc, RETRO))
    doc_path = os.path.join(RETRO, docs[0])
    stem = os.path.splitext(docs[0])[0]

    out_dir = os.path.join(RETRO, stem + "-slides")
    work = os.path.join(out_dir, ".build")
    os.makedirs(work, exist_ok=True)

    with open(CSS, encoding="utf-8") as f:
        css = f.read()
    with open(doc_path, encoding="utf-8") as f:
        units = slides(f.read())

    made = 0
    for num, name, markup in units:
        name = "%s-%s" % (num, name)
        if only and not name.startswith(only):
            continue
        im = shoot(PAGE % (css, markup), work, name)
        im.save(os.path.join(out_dir, name + ".png"))
        print("%-56s %d x %d" % (name + ".png", im.width, im.height))
        made += 1

    for n in os.listdir(work):
        os.remove(os.path.join(work, n))
    os.rmdir(work)
    print("\n%d image%s -> %s" % (made, "" if made == 1 else "s", out_dir))


if __name__ == "__main__":
    main(sys.argv[1:])
