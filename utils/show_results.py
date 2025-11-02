from IPython.display import display, HTML
import pandas as pd
import io
import base64, json

import re


def show_chart_grounding(chart1_img, chart2_img, chart1_gt, chart1_pred, chart2_gt, chart2_pred, chart1_grounding_score, chart2_grounding_score, content_type):
    """
    Display two charts side by side, with ground-truth and predicted grounding tables beneath each.
    chart*_img can be file paths or PIL.Image objects.
    chart*_gt / chart*_pred are CSV (or JSON) strings.
    And Grounding Scores.
    """

    def img_to_base64(img):
        if isinstance(img, str):  # file path
            with open(img, "rb") as f:
                data = f.read()
        else:  # PIL Image
            from io import BytesIO
            buf = BytesIO()
            img.save(buf, format="PNG")
            data = buf.getvalue()
        return base64.b64encode(data).decode("utf-8")
    
    def csv_to_html_table(csv_str):

        # remove brackets (like "million")
        csv_str = re.sub(r'\(.*?\)', '', csv_str).strip()

        csv_str = csv_str.replace("\t", ",")
        csv_str = "\n".join(
            [",".join(col.strip() for col in line.split(",")) for line in csv_str.strip().splitlines() if line.strip()]
        )
        df = pd.read_csv(io.StringIO(csv_str), dtype=str)
        return df.to_html(index=False, escape=False, border=1)

    def format_json(data):
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                return f"<pre>{data}</pre>"
        formatted = json.dumps(data, indent=2, ensure_ascii=False)
        return f"<pre style='background:#f8f8f8; padding:8px; border-radius:6px;'>{formatted}</pre>"

    
    chart1_gt_to_display = csv_to_html_table(chart1_gt) if content_type == "data" else format_json(chart1_gt)
    chart1_pred_to_display = csv_to_html_table(chart1_pred) if content_type == "data" else format_json(chart1_pred)
    
    chart2_gt_to_display = csv_to_html_table(chart2_gt) if content_type == "data" else format_json(chart2_gt)
    chart2_pred_to_display = csv_to_html_table(chart2_pred) if content_type == "data" else format_json(chart2_pred)
    

    chart1_grounding_score_text_to_display = None
    chart2_grounding_score_text_to_display = None

    if content_type == "data":
        chart1_grounding_score_text_to_display = f"Grounding Metric (CSV Precision) = {chart1_grounding_score}"
        chart2_grounding_score_text_to_display = f"Grounding Metric (CSV Precision) = {chart2_grounding_score}"
    elif content_type == "legend":
        chart1_grounding_score_text_to_display = ""
        chart2_grounding_score_text_to_display = ""
    elif content_type == "color":
        chart1_grounding_score_text_to_display = f"Grounding Metric (RGB L2 distance) = {chart1_grounding_score}"
        chart2_grounding_score_text_to_display = f"Grounding Metric (RGB L2 distance) = {chart2_grounding_score}"
    elif content_type == "text_style":
        chart1_grounding_score_text_to_display = f"Grounding Metric (Accuracy) = {chart1_grounding_score}"
        chart2_grounding_score_text_to_display = f"Grounding Metric (Accuracy) = {chart2_grounding_score}"
    else:
        chart1_grounding_score_text_to_display = "ERROR"
        chart2_grounding_score_text_to_display = "ERROR"


    # Convert images to base64 HTML <img>
    chart1_b64 = img_to_base64(chart1_img)
    chart2_b64 = img_to_base64(chart2_img)

    chart1_html = f"""
    <div style="text-align:left; width:48%; display:inline-block; vertical-align:top;">
        <h3 >Chart 1</h3>
        <img src="data:image/png;base64,{chart1_b64}" style="max-width:100%; border:1px solid #ccc; border-radius:8px;"><br>
        <h4 >Ground Truth Grounding</h4>
        {chart1_gt_to_display}
        <h4 >Predicted Grounding</h4>
        {chart1_pred_to_display}
        <h4 >{chart1_grounding_score_text_to_display}</h4>
    </div>
    """

    chart2_html = f"""
    <div style="text-align:left; width:48%; display:inline-block; vertical-align:top;">
        <h3 >Chart 2</h3>
        <img src="data:image/png;base64,{chart2_b64}" style="max-width:100%; border:1px solid #ccc; border-radius:8px;"><br>
        <h4 >Ground Truth Grounding</h4>
        {chart2_gt_to_display}
        <h4 >Predicted Grounding</h4>
        {chart2_pred_to_display}
        <h4 >{chart2_grounding_score_text_to_display}</h4>
    </div>
    """

    html = f"""
    <div style="display:flex; justify-content:space-between; gap:2%; margin-top:10px;">
        {chart1_html}
        {chart2_html}
    </div>
    """

    display(HTML(html))


def show_chart_alignment(
    chart1_img, chart2_img,
    gt_json, pred_json,
    alignment_score, 
    gt_title="Ground Truth Alignment",
    pred_title="Predicted Alignment",
):
    """
    Display 2 charts horizontally and corresponding JSON below.
    And Alignment Score.
    """

    def img_to_base64(img):
        if isinstance(img, str):  # file path
            with open(img, "rb") as f:
                data = f.read()
        else:  # PIL.Image
            from io import BytesIO
            buf = BytesIO()
            img.save(buf, format="PNG")
            data = buf.getvalue()
        return base64.b64encode(data).decode("utf-8")

    def format_json(data):
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                return f"<pre>{data}</pre>"
        formatted = json.dumps(data, indent=2, ensure_ascii=False)
        return f"<pre style='background:#f8f8f8; padding:8px; border-radius:6px;'>{formatted}</pre>"

    # Convert images
    chart1_b64 = img_to_base64(chart1_img)
    chart2_b64 = img_to_base64(chart2_img)

    gt_html = format_json(gt_json)
    pred_html = format_json(pred_json)

    alignment_score_text = f"Alignment Score = {alignment_score}"
    
    # Layout
    html = f"""
    <div style="display:flex; justify-content:space-between; gap:2%; margin-top:10px;">
        <div style="width:48%; text-align:center;">
            <h3 >Chart 1</h3>
            <img src="data:image/png;base64,{chart1_b64}" style="max-width:100%; border:1px solid #ccc; border-radius:8px;">
        </div>
        <div style="width:48%; text-align:center;">
            <h3 >Chart 2</h3>
            <img src="data:image/png;base64,{chart2_b64}" style="max-width:100%; border:1px solid #ccc; border-radius:8px;">
        </div>
    </div>
    <div style="margin-top:10px;">
        <h4 style="text-align:left; margin:4px 0;">{gt_title}</h4>
        {gt_html}
        <h4 style="text-align:left; margin:8px 0;">{pred_title}</h4>
        {pred_html}
        <h4 style="text-align:left; margin:8px 0;">{alignment_score_text}</h4>
    </div>
    """

    display(HTML(html))
