# Regenerates tokyo_weather.pdf, the fixture used by the PDF tests. The deliberately
# odd 45 C is what assert_weather checks, proving the model read the file rather than
# guessing a plausible temperature.  Run: python testfiles/make_tokyo_pdf.py

import os

LINES = [
    (18, "Tokyo Weather Report"),
    (12, "City: Tokyo"),
    (12, "Temperature: 45 C"),
    (12, "Conditions: sunny, humid, hazy"),
    (12, "Jacket recommended: no"),
]


def _stream():
    out = ["BT", "72 720 Td"]
    first = True
    for size, text in LINES:
        out.append("/F1 %d Tf" % size)
        if not first:
            out.append("0 -28 Td")
        first = False
        out.append("(%s) Tj" % text.replace("(", r"\(").replace(")", r"\)"))
    out.append("ET")
    return "\n".join(out)


def build():
    content = _stream().encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    buf = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(buf))
        buf += b"%d 0 obj\n%s\nendobj\n" % (i, body)
    xref_pos = len(buf)
    buf += b"xref\n0 %d\n" % (len(objs) + 1)
    buf += b"0000000000 65535 f \n"
    for off in offsets:
        buf += b"%010d 00000 n \n" % off
    buf += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objs) + 1)
    buf += b"startxref\n%d\n%%%%EOF\n" % xref_pos
    return buf


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "tokyo_weather.pdf")
    with open(path, "wb") as f:
        f.write(build())
    print("wrote", path, os.path.getsize(path), "bytes")
