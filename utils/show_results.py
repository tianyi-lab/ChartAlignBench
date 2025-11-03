from IPython.display import display, HTML

import pandas as pd
import numpy as np

import io
from io import BytesIO

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


def show_chart_robustness(ds_with_images, predictions, scores, gt_alignment, mean_score, std_score):
    html_content = f"""
    <style>
        .container {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .pair-container {{ border: 2px solid #ddd; margin: 20px 0; padding: 15px; border-radius: 8px; }}
        .image-pair {{ display: flex; gap: 20px; margin: 15px 0; }}
        .image-pair img {{ max-width: 45%; height: auto; }}
        .json-box {{ background: #f9f9f9; padding: 10px; border-left: 3px solid #4CAF50; margin: 10px 0; font-family: monospace;}}
        .score {{ font-size: 18px; font-weight: bold; color: #2196F3; }}
        .gt-box {{ background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; font-family: monospace;}}
    </style>
    
    <div class="container">
        <h1>Alignment Evaluation Results</h1>
        
        <div class="summary">
            <h4><strong>Robustness Score (Data Alignment for set of chart pairs over Attribute Variation): </strong> <span class="score">{std_score:.2f}</span></h4>
        </div>
        
        <div class="gt-box">
            <h2>Ground Truth Alignment</h2>
            <pre>{json.dumps(gt_alignment, indent=2)}</pre>
        </div>
    """
    
    for i in range(len(ds_with_images)):
        # Convert image to base64 for embedding
        img = ds_with_images[i]['image_pair']
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        html_content += f"""
        <div class="pair-container">
            <h2>Image Pair {i+1}</h2>
            
            <div class="image-pair">
                <img src="data:image/png;base64,{img_str}" style="max-width:100%; border:1px solid #ccc; border-radius:8px;" alt="Image Pair {i+1}">
            </div>
            
            <div class="json-box">
                <h3>Prediction Alignment</h3>
                <pre>{json.dumps(predictions[i], indent=2)}</pre>
            </div>
            
            <p><strong>Data Alignment Score:</strong> <span class="score">{scores[i]:.2f}</span></p>
        </div>
        """
    
    html_content += "</div>"
    display(HTML(html_content))