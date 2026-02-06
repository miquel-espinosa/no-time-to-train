import json
import re
from pathlib import Path

DATA_ROOT = Path("./data")
IGNORE_DIR = "ignore_datasets_atm"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def extract_bibtex_info(info_path):
    if not info_path.exists():
        return [], None, "--"

    text = info_path.read_text()

    bib_entries = re.findall(r"@[\w]+\{([^,]+),([\s\S]*?)\n\}", text)
    keys = []
    years = []

    for key, body in bib_entries:
        keys.append(key.strip())
        year_match = re.search(r"year\s*=\s*[{\"](\d{4})[}\"]", body)
        if year_match:
            years.append(int(year_match.group(1)))

    year = min(years) if years else None

    res_match = re.search(
        r"(?:resolution|gsd)\s*[:=]?\s*([0-9.]+(?:\s*[-–]\s*[0-9.]+)?\s*(?:m|cm)?)",
        text,
        flags=re.IGNORECASE,
    )
    resolution = (
        re.sub(r"\s*([-–])\s*", r"\1", res_match.group(1).strip())
        if res_match else "--"
    )


    return keys, year, resolution

def summarize_dataset(ds_path):
    ann = ds_path / "annotations"
    train_json = ann / "train.json"
    test_json = ann / "test.json"

    if not train_json.exists() or not test_json.exists():
        return None

    train = load_json(train_json)
    test = load_json(test_json)

    num_categories = len({c["id"] for c in train.get("categories", [])})

    train_images = {i["id"]: i for i in train.get("images", [])}
    test_images = {i["id"]: i for i in test.get("images", [])}

    n_train_img = len(train_images)
    n_test_img = len(test_images)

    n_train_inst = len(train.get("annotations", []))
    n_test_inst = len(test.get("annotations", []))

    total_img = n_train_img + n_test_img
    total_inst = n_train_inst + n_test_inst
    mean_inst = round(total_inst / total_img, 2) if total_img else "--"

    widths = [
        img["width"]
        for img in list(train_images.values()) + list(test_images.values())
        if "width" in img
    ]
    if widths:
        wmin, wmax = min(widths), max(widths)
        width_str = str(wmin) if wmin == wmax else f"{wmin}--{wmax}"
    else:
        width_str = "--"

    sam_masks = (
        (ann / "train_bbox.json").exists()
        or (ann / "test_bbox.json").exists()
    )
    sam_str = r"\cmark" if sam_masks else r"\xmark"

    bib_keys, year, resolution = extract_bibtex_info(ds_path / "info.txt")
    cite = r"~\citep{" + ",".join(bib_keys) + "}" if bib_keys else ""

    return {
        "name": ds_path.name,
        "cite": cite,
        "categories": num_categories,
        "train_img": n_train_img,
        "test_img": n_test_img,
        "train_inst": n_train_inst,
        "test_inst": n_test_inst,
        "mean_inst": mean_inst,
        "sam": sam_str,
        "resolution": resolution,
        "width": width_str,
        "year": year,
    }

def latex_escape(s):
    return s.replace("_", r"\_")

def sort_key(r):
    return (r["year"] is None, r["year"] or 9999)

def main():
    rows = []

    for ds in DATA_ROOT.iterdir():
        if not ds.is_dir() or ds.name == IGNORE_DIR:
            continue
        s = summarize_dataset(ds)
        if s:
            rows.append(s)

    rows.sort(key=sort_key)

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\begin{adjustbox}{width=\linewidth}")
    print(r"\addtolength{\tabcolsep}{-0.4em}")
    print(r"\begin{tabular}{lcccccccccc}")
    print(r"\toprule")
    print(
        r"\textbf{Dataset} & \textbf{\#Cat.} & "
        r"\textbf{Train imgs} & \textbf{Test imgs} & "
        r"\textbf{Train inst.} & \textbf{Test inst.} & "
        r"\textbf{Inst./Img.} & \textbf{SAM} & "
        r"\textbf{Res. (m)} & \textbf{Width} & \textbf{Year} \\"
    )
    print(r"\midrule")

    for r in rows:
        year = r["year"] if r["year"] is not None else "--"
        print(
            f"{latex_escape(r['name'])}{r['cite']} & "
            f"{r['categories']} & "
            f"{r['train_img']} & "
            f"{r['test_img']} & "
            f"{r['train_inst']} & "
            f"{r['test_inst']} & "
            f"{r['mean_inst']} & "
            f"{r['sam']} & "
            f"{r['resolution']} & "
            f"{r['width']} & "
            f"{year} \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{adjustbox}")
    print(r"\caption{Summary of EO datasets used in our experiments.}")
    print(r"\label{tab:datasets}")
    print(r"\end{table}")

if __name__ == "__main__":
    main()
