#!/usr/bin/env python3
"""Editable SVG master + vector PDF from one small scene description.

Run: python figure_scripts/fig1.py
Requires reportlab for PDF export; the SVG itself uses only standard SVG elements.
Coordinates are in a 1200 x 1290 design space; the PDF is 5.5 inches wide.
"""

from __future__ import annotations

import argparse
import base64
import math
from pathlib import Path
from xml.etree import ElementTree as ET

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
W, H = 1200, 1290
C = {
    "ink": "#202020",
    "muted": "#666666",
    "edge": "#444444",
    "border": "#D1D1D1",
    "pale": "#FAFAFA",
    "panel_fill": "#F0F0F0",
    "highlight": "#E6E6E6",
    "accent": "#315F83",
    "white": "#FFFFFF",
}

FONT_FILES = {
    "text": "Times New Roman.ttf",
    "bold": "Times New Roman Bold.ttf",
    "italic": "Times New Roman Italic.ttf",
    "bolditalic": "Times New Roman Bold Italic.ttf",
    "math": "RHMCMItalic.ttf",
    "roman": "RHMCMRoman.ttf",
    "symbol": "RHMCMSymbols.ttf",
}
SVG_FONTS = {
    "text": ('"Times New Roman", Times, serif', "normal", "normal"),
    "bold": ('"Times New Roman", Times, serif', "bold", "normal"),
    "italic": ('"Times New Roman", Times, serif', "normal", "italic"),
    "bolditalic": ('"Times New Roman", Times, serif', "bold", "italic"),
    "math": ('"RHM CM Math", serif', "normal", "normal"),
    "roman": ('"RHM CM Roman", serif', "normal", "normal"),
    "symbol": ('"RHM CM Symbols", serif', "normal", "normal"),
}


def register_fonts(font_dir: Path) -> None:
    for name, filename in FONT_FILES.items():
        path = (HERE / "fonts" if name in {"math", "roman", "symbol"} else font_dir) / filename
        if not path.exists() and name in {"text", "bold", "italic", "bolditalic"}:
            # PDF's built-in Times fonts provide a portable fallback on Linux.
            base = {"text": "Times-Roman", "bold": "Times-Bold",
                    "italic": "Times-Italic", "bolditalic": "Times-BoldItalic"}[name]
            pdfmetrics.registerFont(pdfmetrics.Font(name, base, "WinAnsiEncoding"))
        elif path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path)))
        else:
            raise FileNotFoundError(f"Missing mathematical font asset: {path}")


def font_runs(text, font):
    """Use Computer Modern letters, upright numbers, and TeX math symbols."""
    if font != "math":
        return [(text, font)]
    runs = []
    for char in text:
        candidates = ("math", "roman", "symbol") if char.isalpha() else ("roman", "symbol", "math")
        selected = next((name for name in candidates if ord(char) in pdfmetrics.getFont(name).face.charToGlyph), "text")
        if runs and runs[-1][1] == selected:
            runs[-1] = (runs[-1][0] + char, selected)
        else:
            runs.append((char, selected))
    return runs


def text_width(text, font, size):
    return sum(pdfmetrics.stringWidth(t, f, size) for t, f in font_runs(text, font))


def number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


class Scene:
    def __init__(self):
        self.layers = []
        self.items = None

    def layer(self, name: str, label: str):
        self.items = []
        self.layers.append((name, label, self.items))

    def add(self, kind, **args):
        self.items.append((kind, args))

    def rect(self, x, y, w, h, fill="none", stroke="none", sw=1, r=0):
        self.add("rect", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, sw=sw, r=r)

    def line(self, x1, y1, x2, y2, color=None, sw=1.8):
        self.add("line", x1=x1, y1=y1, x2=x2, y2=y2, stroke=color or C["edge"], sw=sw)

    def circle(self, x, y, radius, fill=None, stroke=None, sw=1.8):
        self.add(
            "circle",
            x=x,
            y=y,
            radius=radius,
            fill=fill or C["white"],
            stroke=stroke or C["edge"],
            sw=sw,
        )

    def path(self, commands, fill="none", stroke="none", sw=1.8):
        self.add("path", commands=commands, fill=fill, stroke=stroke, sw=sw)

    def text(self, x, y, text, size=22, font="text", color=None, anchor="start"):
        if font == "math":
            width = text_width(text, font, size)
            x -= width / 2 if anchor == "middle" else width if anchor == "end" else 0
            for part, actual_font in font_runs(text, font):
                self.add("text", x=x, y=y, text=part, size=size, font=actual_font,
                         fill=color or C["ink"], anchor="start")
                x += pdfmetrics.stringWidth(part, actual_font, size)
            return
        self.add(
            "text",
            x=x,
            y=y,
            text=text,
            size=size,
            font=font,
            fill=color or C["ink"],
            anchor=anchor,
        )

    def rich_text(self, x, y, runs, size=22, color=None, anchor="start"):
        """Mix ordinary words and mathematical variables as editable text."""
        width = sum(text_width(t, f, size) for t, f in runs)
        x -= width / 2 if anchor == "middle" else width if anchor == "end" else 0
        for text, font in runs:
            self.text(x, y, text, size, font, color)
            x += text_width(text, font, size)

    def arrow(self, x1, y1, x2, y2, color=None, sw=2, head=8):
        color = color or C["muted"]
        self.line(x1, y1, x2, y2, color, sw)
        theta = math.atan2(y2 - y1, x2 - x1)
        bx, by = x2 - head * math.cos(theta), y2 - head * math.sin(theta)
        dx, dy = head * 0.48 * math.sin(theta), -head * 0.48 * math.cos(theta)
        self.path(
            [("M", x2, y2), ("L", bx + dx, by + dy), ("L", bx - dx, by - dy), ("Z",)],
            fill=color,
        )

    def math_power(self, x, y, prefix, exponent, size=32, color=None, anchor="start"):
        """Separate, editable live text for a normal base and its superscript."""
        color = color or C["ink"]
        # Match LaTeX: italic variables, upright exponents, and CMSY relations.
        segments = []
        for i, part in enumerate(prefix.split("∼")):
            if i:
                segments.append(("∼", "symbol"))
            segments.append((part, "math"))
        base_width = sum(text_width(t, f, size) for t, f in segments)
        exp_width = text_width(exponent, "math", size * 0.64)
        width = base_width + exp_width + 1
        x -= width / 2 if anchor == "middle" else width if anchor == "end" else 0
        cursor = x
        for t, f in segments:
            self.text(cursor, y, t, size, f, color)
            cursor += text_width(t, f, size)
        self.text(
            x + base_width + 1, y - size * 0.43, exponent, size * 0.64, "math", color
        )

    def panel(self, x, y, w, h, title):
        self.rect(x, y, w, h, C["white"], C["border"], 1.2, 10)
        self.rect(x + 1, y + 1, w - 2, 45, C["panel_fill"], r=9)
        self.rect(x + 1, y + 24, w - 2, 22, C["panel_fill"])
        self.text(x + 20, y + 31, title, 25, "bold", C["ink"])

    def node(self, x, y, label="", kind="latent", accent=None):
        if kind == "target":
            self.rect(x - 16, y - 17, 32, 34, C["white"], accent or C["ink"], 2.8, 3)
            self.text(x, y + 8, label or "Z", 26, "math", C["ink"], "middle")
        elif kind == "token":
            self.text(x, y + 8, label, 25, "roman", C["ink"], "middle")
        else:
            radius = 16 if label == "X" else 12
            self.circle(
                x,
                y,
                radius,
                C["highlight"] if label == "X" else C["white"],
                C["ink"] if label == "X" else C["edge"],
            )
            if label:
                self.text(
                    x,
                    y + 7.5,
                    label,
                    24 if label == "X" else 23,
                    "math",
                    C["ink"],
                    "middle",
                )

    def cluster(self, parent, children):
        px, py = parent
        xs = [p[0] for p in children]
        cy = children[0][1]
        self.path(
            [
                ("M", px - 18, py - 21),
                ("Q", px, py - 28, px + 18, py - 21),
                ("L", max(xs) + 28, cy + 5),
                ("Q", max(xs) + 34, cy + 23, max(xs) + 16, cy + 23),
                ("L", min(xs) - 16, cy + 23),
                ("Q", min(xs) - 34, cy + 23, min(xs) - 28, cy + 5),
                ("Z",),
            ],
            fill=C["highlight"],
        )

    def collapsed(self, x, y, bottom, half=27, *, lower_dots=False):
        self.path(
            [
                ("M", x, y + 13),
                ("L", x - half, bottom - 2),
                ("L", x + half, bottom - 2),
                ("Z",),
            ],
            C["pale"],
            C["border"],
            1.2,
        )
        # Baseline dots rather than centered dots; the small continuations in
        # panel d put them below the abbreviated subtree, as in \ldots.
        self.text(x, bottom + 13 if lower_dots else bottom - 9,
                  "...", 16 if lower_dots else 20, "roman", C["muted"], "middle")

    def tree(
        self,
        nodes,
        rules,
        critical,
        accent,
        *,
        target=None,
        tuple_nodes=(),
        tuple_label=None,
    ):
        """Rule = (id, parent, children). Critical maps rule id to path segments.

        A segment is 'parent' or a child node id. One filled square = one
        unresolved production rule, including a sibling-to-sibling traversal.
        """
        for rule_id, parent, children in rules:
            px, py, *_ = nodes[parent]
            ry = py + 24
            path_segments = critical.get(rule_id, ())
            color = accent if "parent" in path_segments else C["edge"]
            self.line(
                px, py + 12, px, ry, color, 3 if "parent" in path_segments else 1.8
            )
            for child in children:
                cx, cy, label, kind = nodes[child]
                active = child in path_segments
                endpoint = cy - 14 if kind != "token" else cy - 13
                self.line(
                    px,
                    ry,
                    cx,
                    endpoint,
                    accent if active else C["edge"],
                    3 if active else 1.8,
                )
            active = rule_id in critical
            self.rect(
                px - 4.5,
                ry - 4.5,
                9,
                9,
                accent if active else C["white"],
                accent if active else C["edge"],
                1.6,
            )
        for node_id, (x, y, label, kind) in nodes.items():
            self.node(
                x,
                y,
                label,
                "target" if node_id == target else kind,
                accent if node_id == target else None,
            )
        if tuple_label:
            xs = [nodes[k][0] for k in tuple_nodes]
            cy = nodes[tuple_nodes[0]][1]
            self.text(
                min(xs) - 37, cy + 8, tuple_label, 25, "math", C["ink"], "middle"
            )

    def svg(self, path):
        ns = "http://www.w3.org/2000/svg"
        # Inkscape's official namespace URI is intentionally not an https URL.
        ink = "http://www.inkscape.org/namespaces/inkscape"
        ET.register_namespace("", ns)
        ET.register_namespace("inkscape", ink)
        root = ET.Element(
            f"{{{ns}}}svg",
            {
                "width": "5.5in",
                "height": f"{H / W * 5.5:g}in",
                "viewBox": f"0 0 {W} {H}",
                "version": "1.1",
                "role": "img",
                "aria-labelledby": "figure-title figure-description",
            },
        )
        ET.SubElement(
            root, f"{{{ns}}}title", {"id": "figure-title"}
        ).text = "Learn the hierarchy by predicting nearby latents"
        ET.SubElement(root, f"{{{ns}}}desc", {"id": "figure-description"}).text = (
            "A: A Random Hierarchy Model grammar generates visible tokens from latent symbols; an example sample appears on the right. "
            "Trained machines learn the inverse map by clustering synonymous tuples into latents. "
            "The middle path numbers production-rule squares to define the distance r. "
            "B and C: Root classification and token prediction require long correlation paths, costing samples exponential in tree depth. "
            "D: Predicting cousin latents keeps the path at three production rules at every level, yielding order v m cubed samples up to logarithms."
        )
        ET.SubElement(
            root, f"{{{ns}}}metadata"
        ).text = "Draft 4. Times text and Computer Modern mathematics; one accent colour for correlation paths. Live editable text; named layers. Generated by generate.py. No raster elements."
        # Embed the small, redistributable math fonts to keep the SVG portable.
        # Illustrator can also use the same TTFs from fonts/ via Font Book.
        defs = ET.SubElement(root, f"{{{ns}}}defs")
        style = ET.SubElement(defs, f"{{{ns}}}style", {"type": "text/css"})
        style.text = "\n".join(
            '@font-face {font-family:' + SVG_FONTS[key][0].split(",")[0] +
            ';src:url(data:font/ttf;base64,' +
            base64.b64encode((HERE / "fonts" / FONT_FILES[key]).read_bytes()).decode("ascii") +
            ') format("truetype");font-weight:normal;font-style:normal;}'
            for key in ["math", "roman", "symbol"]
        )
        for layer_id, label, items in self.layers:
            group = ET.SubElement(
                root,
                f"{{{ns}}}g",
                {
                    "id": layer_id,
                    f"{{{ink}}}groupmode": "layer",
                    f"{{{ink}}}label": label,
                },
            )
            for index, (kind, a) in enumerate(items):
                attr = {"id": f"{layer_id}-{kind}-{index:03d}"}
                if kind == "rect":
                    attr.update(
                        x=number(a["x"]),
                        y=number(a["y"]),
                        width=number(a["w"]),
                        height=number(a["h"]),
                        rx=number(a["r"]),
                    )
                elif kind == "circle":
                    attr.update(
                        cx=number(a["x"]), cy=number(a["y"]), r=number(a["radius"])
                    )
                elif kind == "line":
                    attr.update({k: number(a[k]) for k in ["x1", "y1", "x2", "y2"]})
                elif kind == "path":
                    attr["d"] = " ".join(
                        c[0] + " " + " ".join(number(v) for v in c[1:])
                        for c in a["commands"]
                    )
                elif kind == "text":
                    family, weight, style = SVG_FONTS[a["font"]]
                    attr.update(
                        x=number(a["x"]),
                        y=number(a["y"]),
                        **{
                            "font-family": family,
                            "font-size": number(a["size"]),
                            "font-weight": weight,
                            "font-style": style,
                            "text-anchor": a["anchor"],
                            "{http://www.w3.org/XML/1998/namespace}space": "preserve",
                        },
                    )
                attr["fill"] = a.get("fill", "none")
                if "stroke" in a:
                    attr.update(
                        stroke=a["stroke"],
                        **{
                            "stroke-width": number(a["sw"]),
                            "stroke-linejoin": "round",
                            "stroke-linecap": "round",
                        },
                    )
                element = ET.SubElement(group, f"{{{ns}}}{kind}", attr)
                if kind == "text":
                    element.text = a["text"]
        ET.indent(root)
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    def pdf(self, path):
        scale = 5.5 * 72 / W
        pdf = canvas.Canvas(
            str(path), pagesize=(W * scale, H * scale), pageCompression=1, invariant=1
        )
        pdf.setTitle("Figure 1 draft: learn from nearby latents")
        pdf.setAuthor("Daniel J. Korchinski et al.")
        pdf.setSubject(
            "RHM, synonym clustering, correlation distance, and sample complexity"
        )
        pdf.scale(scale, scale)
        for _, _, items in self.layers:
            for kind, a in items:
                fill, stroke = a.get("fill", "none"), a.get("stroke", "none")
                if fill != "none":
                    pdf.setFillColor(fill)
                if stroke != "none":
                    pdf.setStrokeColor(stroke)
                    pdf.setLineWidth(a["sw"])
                    pdf.setLineJoin(1)
                    pdf.setLineCap(1)
                paint = {"fill": int(fill != "none"), "stroke": int(stroke != "none")}
                if kind == "rect":
                    pdf.roundRect(
                        a["x"], H - a["y"] - a["h"], a["w"], a["h"], a["r"], **paint
                    )
                elif kind == "circle":
                    pdf.circle(a["x"], H - a["y"], a["radius"], **paint)
                elif kind == "line":
                    pdf.line(a["x1"], H - a["y1"], a["x2"], H - a["y2"])
                elif kind == "path":
                    p = pdf.beginPath()
                    last = (0, 0)
                    for command, *v in a["commands"]:
                        if command == "M":
                            p.moveTo(v[0], H - v[1])
                            last = tuple(v)
                        elif command == "L":
                            p.lineTo(v[0], H - v[1])
                            last = tuple(v)
                        elif command == "Q":
                            x0, y0 = last
                            qx, qy, x1, y1 = v
                            p.curveTo(
                                x0 + 2 / 3 * (qx - x0),
                                H - (y0 + 2 / 3 * (qy - y0)),
                                x1 + 2 / 3 * (qx - x1),
                                H - (y1 + 2 / 3 * (qy - y1)),
                                x1,
                                H - y1,
                            )
                            last = (x1, y1)
                        elif command == "Z":
                            p.close()
                    pdf.drawPath(p, **paint)
                elif kind == "text":
                    pdf.setFont(a["font"], a["size"])
                    draw = {
                        "start": pdf.drawString,
                        "middle": pdf.drawCentredString,
                        "end": pdf.drawRightString,
                    }[a["anchor"]]
                    draw(a["x"], H - a["y"], a["text"])
        pdf.showPage()
        pdf.save()


def draw_rhm(s):
    s.layer("a-rhm", "a | Data: the Random Hierarchy Model")
    s.panel(
        20,
        16,
        1160,
        284,
        "a   Random Hierarchy Model (RHM)",
    )
    s.rich_text(42, 94, [("Each symbol has ", "text"), ("m", "math"),
                        (" production rules.", "text")], 23)
    # Center the table in the left content area (x=42..590, y=110..248).
    # Use the actual column widths so asymmetric rules do not shift its center.
    columns = [("α → {aab, aba}", "a → {000, 101}"),
               ("β → {bba, bab}", "b → {111, 011}")]
    widths = [max(text_width(rule, "math", 28) for rule in column)
              for column in columns]
    column_gap = 60
    cursor = 316 - (sum(widths) + column_gap) / 2
    for column, width in zip(columns, widths):
        for y, rule in zip((164, 208), column):
            s.text(cursor, y, rule, 28, "math", C["ink"])
        cursor += width + column_gap
    caption_y = 277
    s.rich_text(316, caption_y, [("RHM grammar parameters: ", "text"),
                          ("L = 2,  v = 2,  m = 2,  s = 3", "math")],
                22, anchor="middle")
    nodes = {"root": (887, 96, "α", "latent")}
    rules = [("r0", "root", ["a0", "b0", "a1"])]
    for name, x, label, digits in [
        ("a0", 762, "a", "000"),
        ("b0", 887, "b", "111"),
        ("a1", 1012, "a", "101"),
    ]:
        nodes[name] = (x, 158, label, "latent")
        children = []
        for j, digit in enumerate(digits):
            child = f"{name}-{j}"
            nodes[child] = (x + (j - 1) * 22, 222, digit, "token")
            children.append(child)
        rules.append((f"r-{name}", name, children))
    s.tree(nodes, rules, {}, C["ink"])
    s.text(644, 164, "latent", 20, color=C["muted"])
    s.text(644, 228, "visible", 20, color=C["muted"])
    s.text(1084, 127, "Generate samples", 20, color=C["ink"], anchor="middle")
    s.text(1084, 151, "top-down", 20, color=C["ink"], anchor="middle")
    s.arrow(1084, 164, 1084, 218, C["ink"], 1.8, 8)
    s.text(887, caption_y, "Example RHM sample", 23, anchor="middle")


def draw_mechanism(s):
    s.layer("mechanism", "Mechanism | Rule distance attenuates the signal")
    # Equal leading, with matching visible gaps to panel a and panels b/c.
    headline_y, line_gap = 403, 48
    s.rich_text(
        40, headline_y - line_gap,
        [("Trained machines must learn the inverse map, clustering tuples into latents, e.g. ", "text"),
         ("{aab, aba} → α", "math"), (".", "text")],
        23,
    )
    s.rich_text(
        40,
        headline_y,
        [("The training signal to cluster weakens with distance ", "bold"),
         ("r", "math"), (" to target", "bold")],
        26,
    )
    s.rich_text(
        40, headline_y + line_gap,
        [("Each unresolved rule adds a factor ", "text"), ("m", "math"),
         (" to the samples needed, ", "text"), ("P", "math"), (".", "text")],
        23,
    )
    # Center the path and formula together in the white space to the right
    # of the two explanatory lines. Index production-rule squares, not edges.
    dx, dy = -28, -12
    s.text(801 + dx, 429 + dy, "T", 25, "math", C["ink"], "middle")
    s.line(817 + dx, 421 + dy, 917 + dx, 421 + dy, C["accent"], 3)
    s.line(962 + dx, 421 + dy, 1010 + dx, 421 + dy, C["accent"], 3)
    for x, label in [(840, "1"), (892, "2"), (988, "r")]:
        s.rect(x + dx - 5, 416 + dy, 10, 10, C["accent"])
        s.text(x + dx, 457 + dy, label, 23, "math" if label == "r" else "roman", C["ink"], "middle")
    s.text(940 + dx, 427 + dy, "···", 24, color=C["ink"], anchor="middle")
    s.node(1028 + dx, 421 + dy, "Z", "target", C["accent"])
    s.math_power(1118 + dx, 435 + dy, "P ∼ m", "r", 35, C["ink"], "middle")


def draw_cost_result(s, center, y, distance, exponent, description):
    """Center the distance-to-cost relation above its sample-cost sentence."""
    s.math_power(center, y, f"r = {distance}  ⇒  P ∼ m", exponent,
                 30, C["ink"], "middle")
    s.rich_text(center, y + 33,
                [(description, "bold"), ("exp(", "bold"), ("L", "bolditalic"),
                 (") samples.", "bold")],
                22, C["ink"], "middle")


def draw_classification(s):
    s.layer("b-classification", "b | Tokens to latents: supervised classification")
    s.panel(20, 495, 570, 366, "b   Supervised classification")
    nodes = {
        "root": (250, 604, "Z", "latent"),
        "u0": (126, 662, "", "latent"),
        "u1": (250, 662, "", "latent"),
        "u2": (374, 662, "", "latent"),
        "v0": (330, 720, "", "latent"),
        "v1": (374, 720, "", "latent"),
        "x": (418, 720, "X", "latent"),
        "t0": (393, 779, "1", "token"),
        "t1": (418, 779, "0", "token"),
        "t2": (443, 779, "1", "token"),
    }
    nodes = {key: (x + 22, y - 24, label, kind) for key, (x, y, label, kind) in nodes.items()}
    for node in ["u0", "u1", "v0", "v1"]:
        x, y, *_ = nodes[node]
        s.collapsed(x, y, 765, 31 if node.startswith("u") else 17)
    s.cluster(nodes["x"][:2], [nodes[t][:2] for t in ["t0", "t1", "t2"]])
    rules = [
        ("r0", "root", ["u0", "u1", "u2"]),
        ("r1", "u2", ["v0", "v1", "x"]),
        ("r2", "x", ["t0", "t1", "t2"]),
    ]
    critical = {
        "r0": ["parent", "u2"],
        "r1": ["parent", "x"],
        "r2": ["parent", "t0", "t1", "t2"],
    }
    s.tree(
        nodes,
        rules,
        critical,
        C["accent"],
        target="root",
        tuple_nodes=["t0", "t1", "t2"],
        tuple_label="T",
    )
    draw_cost_result(s, 305, 812, "L", "L",
                     "Predicting the root from tokens costs ")


def draw_token_prediction(s):
    s.layer("c-token-prediction", "c | Latents to tokens: MLM or diffusion")
    s.panel(
        610,
        495,
        570,
        366,
        "c   Token prediction (MLM, diffusion)",
    )
    nodes = {
        "root": (909, 604, "", "latent"),
        "x": (746, 662, "X", "latent"),
        "mid": (909, 662, "", "latent"),
        "right": (1072, 662, "", "latent"),
        "t0": (704, 720, "a", "latent"),
        "t1": (746, 720, "b", "latent"),
        "t2": (788, 720, "a", "latent"),
        "r0": (1030, 720, "", "latent"),
        "r1": (1072, 720, "", "latent"),
        "r2": (1114, 720, "", "latent"),
        "z0": (1089, 779, "0", "token"),
        "z1": (1114, 779, "1", "token"),
        "z": (1139, 779, "Z", "token"),
    }
    nodes = {key: (x, y - 24, label, kind) for key, (x, y, label, kind) in nodes.items()}
    for node in ["mid", "t0", "t1", "t2", "r0", "r1"]:
        x, y, *_ = nodes[node]
        s.collapsed(x, y, 765, 29 if node == "mid" else 16)
    s.cluster(nodes["x"][:2], [nodes[t][:2] for t in ["t0", "t1", "t2"]])
    rules = [
        ("r0", "root", ["x", "mid", "right"]),
        ("rx", "x", ["t0", "t1", "t2"]),
        ("rr", "right", ["r0", "r1", "r2"]),
        ("rz", "r2", ["z0", "z1", "z"]),
    ]
    critical = {
        "r0": ["x", "right"],
        "rx": ["parent", "t0", "t1", "t2"],
        "rr": ["parent", "r2"],
        "rz": ["parent", "z"],
    }
    s.tree(
        nodes,
        rules,
        critical,
        C["accent"],
        target="z",
        tuple_nodes=["t0", "t1", "t2"],
        tuple_label="T",
    )
    draw_cost_result(s, 895, 812, "L + 1", "L+1",
                     "Predicting tokens from latents costs ")


def draw_local_motif(s, dx, *, upper=False, dy=0):
    def n(x, y, label="", kind="latent"):
        return (x + dx, y + dy, label, kind)

    nodes = {
        "g": n(225, 1020),
        "x": n(126, 1072, "X"),
        "mid": n(225, 1072),
        "right": n(324, 1072),
        "t0": n(93, 1124, "a" if upper else "1", "latent" if upper else "token"),
        "t1": n(126, 1124, "b" if upper else "0", "latent" if upper else "token"),
        "t2": n(159, 1124, "a" if upper else "1", "latent" if upper else "token"),
        "z0": n(291, 1124, "a" if upper else "0", "latent" if upper else "token"),
        "z1": n(324, 1124, "b" if upper else "1", "latent" if upper else "token"),
        "z": n(357, 1124, "Z", "latent" if upper else "token"),
    }
    s.cluster(nodes["x"][:2], [nodes[k][:2] for k in ["t0", "t1", "t2"]])
    s.collapsed(225 + dx, 1072 + dy, 1147 + dy, 17, lower_dots=True)
    if upper:
        for k in ["t0", "t1", "t2", "z0", "z1", "z"]:
            x, y, *_ = nodes[k]
            s.collapsed(x, y, 1164 + dy, 10, lower_dots=True)
    rules = [
        ("rg", "g", ["x", "mid", "right"]),
        ("rx", "x", ["t0", "t1", "t2"]),
        ("rz", "right", ["z0", "z1", "z"]),
    ]
    critical = {
        "rg": ["x", "right"],
        "rx": ["parent", "t0", "t1", "t2"],
        "rz": ["parent", "z"],
    }
    s.tree(
        nodes,
        rules,
        critical,
        C["accent"],
        target="z",
        tuple_nodes=["t0", "t1", "t2"],
        tuple_label="T",
    )


def draw_latent_prediction(s):
    s.layer("d-latent-prediction", "d | Latents to latents: repeat local clustering")
    s.panel(
        20,
        881,
        1160,
        349,
        "d   Predicting latents from latents reduces sample complexity exponentially",
    )
    s.text(
        42,
        959,
        "The target moves up with the representation, keeping the clustering signal strong.",
        24,
    )
    # Previous layout, with titles above the diagrams. To restore it, use
    # diagram_dy = 26 and these four calls instead of the lower titles below:
    # s.text(224, 991, "1   Predict tokens from tokens,", 22, "bold", anchor="middle")
    # s.text(224, 1018, "build latents.", 22, "bold", anchor="middle")
    # s.text(663, 991, "2   Predict latents from latents,", 22, "bold", anchor="middle")
    # s.text(663, 1018, "build super-latents.", 22, "bold", anchor="middle")
    diagram_dy = -24
    draw_local_motif(s, 0, dy=diagram_dy)
    draw_local_motif(s, 438, upper=True, dy=diagram_dy)
    s.arrow(407, 1080 + diagram_dy, 463, 1080 + diagram_dy, C["ink"], 2.5, 9)
    s.text(224, 1178, "1   Predict tokens from tokens,", 22, "bold", anchor="middle")
    s.text(224, 1204, "build latents.", 22, "bold", anchor="middle")
    s.text(663, 1178, "2   Predict latents from latents,", 22, "bold", anchor="middle")
    s.rich_text(663, 1204, [("build ", "bold"), ("super-latents", "bolditalic"),
                          (".", "bold")], 22, anchor="middle")
    s.rect(875, 984, 282, 218, C["panel_fill"], r=8)
    s.rich_text(1016, 1021, [("r = 3", "math"), (" at every level", "bold")], 24, C["ink"], "middle")
    s.math_power(1016, 1079, "P ∼ v m", "3", 44, C["ink"], "middle")
    s.rich_text(1016, 1120, [("Independent of depth ", "bold"), ("L", "math"),
                           ("*", "bold")], 23, C["ink"], "middle")
    s.text(
        1016,
        1172,
        "* Up to logarithmic factors.",
        20,
        color=C["muted"],
        anchor="middle",
    )


def draw_key(s):
    s.layer("key", "Key | Shared visual grammar")
    s.rect(40, 1260, 10, 10, C["muted"])
    s.text(62, 1273, "production rule", 20, color=C["muted"])
    s.rect(247, 1252, 35, 25, C["highlight"], r=5)
    s.rich_text(293, 1273, [("cluster ", "text"), ("T", "math"), (" into ", "text"), ("X", "math")], 20, C["muted"])
    s.line(477, 1267, 511, 1267, C["accent"], 3)
    s.text(523, 1273, "correlation path", 20, color=C["muted"])
    s.rect(717, 1253, 27, 26, C["white"], C["ink"], 1.8, 2)
    s.text(730.5, 1273, "Z", 23, "math", anchor="middle")
    s.text(756, 1273, "prediction target", 20, color=C["muted"])
    s.text(954, 1273, "P: number of samples", 20, color=C["muted"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--font-dir", type=Path, default=Path("/System/Library/Fonts/Supplemental")
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__import__("os").environ.get("FIGURE_OUTPUT", HERE.parent / "figures")))
    args = parser.parse_args()
    register_fonts(args.font_dir)
    s = Scene()
    s.layer("background", "Background")
    s.rect(0, 0, W, H, C["white"])
    for draw in [
        draw_rhm,
        draw_mechanism,
        draw_classification,
        draw_token_prediction,
        draw_latent_prediction,
        draw_key,
    ]:
        draw(s)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    s.svg(args.output_dir / "fig_1_schematic.svg")
    s.pdf(args.output_dir / "fig_1_schematic.pdf")
    print("Wrote editable SVG and vector PDF (5.5 inches wide).")


if __name__ == "__main__":
    main()
