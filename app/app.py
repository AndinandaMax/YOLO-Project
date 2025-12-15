import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Urban AI Inspector",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for futuristic design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(90deg, #00d4ff 0%, #9c27ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
    }
    
    /* File uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.05);
        border: 2px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff 0%, #9c27ff 100%);
        color: white;
        font-family: 'Orbitron', sans-serif;
        font-weight: bold;
        border: none;
        border-radius: 25px;
        padding: 15px 40px;
        font-size: 16px;
        box-shadow: 0 0 20px rgba(156, 39, 255, 0.5);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 40px rgba(156, 39, 255, 0.8);
        transform: translateY(-2px);
    }
    
    /* Metrics */
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        padding: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.2);
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 2px solid rgba(0, 212, 255, 0.3);
    }
    
    /* Custom card */
    .custom-card {
        background: rgba(255, 255, 255, 0.05);
        border: 2px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.1);
    }
    
    /* Glowing text */
    .glow-text {
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.8);
        font-family: 'Orbitron', sans-serif;
        font-weight: bold;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        color: #ffffff;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d4ff 0%, #9c27ff 100%);
    }
</style>
""", unsafe_allow_html=True)

# Class names
CLASS_NAMES = {
    0: "Damaged Road Issues",
    1: "Pothole Issues",
    2: "Illegal Parking Issues",
    3: "Broken Road Sign Issues",
    4: "Fallen Trees",
    5: "Littering/Garbage on Public Places",
    6: "Vandalism Issues",
    7: "Dead Animal Pollution",
    8: "Damaged Concrete Structures",
    9: "Damaged Electric Wires and Poles"
}

# Emoji mapping for each class
CLASS_EMOJI = {
    0: "🛣️", 1: "🕳️", 2: "🚗", 3: "⚠️", 4: "🌳",
    5: "🗑️", 6: "🎨", 7: "💀", 8: "🏗️", 9: "⚡"
}

# Color mapping for bounding boxes
CLASS_COLORS = {
    0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 255),
    5: (0, 255, 255), 6: (128, 0, 128), 7: (255, 128, 0), 8: (0, 128, 255), 9: (128, 255, 0)
}

@st.cache_resource
def load_model(model_path):
    """Load YOLO model with caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def draw_predictions(image, results):
    """Draw bounding boxes on image"""
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Draw bounding box
            color = CLASS_COLORS.get(cls, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with background
            label = f"{CLASS_NAMES[cls]}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            detections.append({
                'class': CLASS_NAMES[cls],
                'confidence': conf,
                'emoji': CLASS_EMOJI[cls]
            })
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, detections

def create_detection_chart(detections):
    """Create interactive chart for detections"""
    if not detections:
        return None
    
    # Count classes
    class_counts = {}
    for det in detections:
        class_name = det['class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=list(class_counts.keys()),
            y=list(class_counts.values()),
            marker=dict(
                color=list(range(len(class_counts))),
                colorscale='Viridis',
                line=dict(color='rgba(0, 212, 255, 0.8)', width=2)
            ),
            text=list(class_counts.values()),
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title="Detection Distribution",
        xaxis_title="Issue Type",
        yaxis_title="Count",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#00d4ff', family='Orbitron'),
        height=400
    )
    
    return fig

# Main App
def main():
    # Header
    st.markdown("""
    <h1 style='text-align: center; font-size: 3.5em; margin-bottom: 0;'>
        🏙️ URBAN AI INSPECTOR
    </h1>
    <p style='text-align: center; color: #00d4ff; font-size: 1.2em; font-family: Rajdhani;'>
        AI-Powered Urban Infrastructure Detection System
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>⚙️ CONTROL PANEL</h2>", unsafe_allow_html=True)
        
        # st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        model_path = st.text_input("🤖 Model Path", value="best.pt", 
                                   help="Path to your YOLO model file")
        confidence_threshold = st.slider("🎯 Confidence Threshold", 
                                        min_value=0.0, max_value=1.0, 
                                        value=0.25, step=0.05)
        # st.markdown("</div>", unsafe_allow_html=True)
        
        # st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<h3>📋 Detectable Issues</h3>", unsafe_allow_html=True)
        for idx, name in CLASS_NAMES.items():
            st.markdown(f"{CLASS_EMOJI[idx]} {name}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<h3>ℹ️ About</h3>", unsafe_allow_html=True)
        st.markdown("""
        This AI system detects various urban infrastructure issues using 
        YOLO object detection.
        
        
        **Model:** YOLOv12n\n
        **Classes:** 10 Urban Issues
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<h2>📤 UPLOAD IMAGE</h2>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of urban infrastructure"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
    
    with col2:
        # st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("<h2>🔍 DETECTION RESULTS</h2>", unsafe_allow_html=True)
        
        if uploaded_file:
            if st.button("🚀 RUN DETECTION", use_container_width=True):
                with st.spinner("🔄 AI Processing..."):
                    # Load model
                    model = load_model(model_path=f"../outputs/models/{model_path}")
                    
                    if model:
                        # Run prediction
                        results = model.predict(
                            source=image,
                            conf=confidence_threshold,
                            save=False
                        )
                        
                        # Draw predictions
                        result_img, detections = draw_predictions(image, results)
                        
                        # Display results
                        st.image(result_img, caption="Detected Issues", use_column_width=True)
                        
                        # Metrics
                        st.markdown("<h3 class='glow-text'>📊 STATISTICS</h3>", unsafe_allow_html=True)
                        metric_cols = st.columns(3)
                        with metric_cols[0]:
                            st.metric("Total Detections", len(detections))
                        with metric_cols[1]:
                            unique_classes = len(set([d['class'] for d in detections]))
                            st.metric("Issue Types", unique_classes)
                        with metric_cols[2]:
                            avg_conf = np.mean([d['confidence'] for d in detections]) if detections else 0
                            st.metric("Avg Confidence", f"{avg_conf:.2%}")
                        
                        # Detection details
                        if detections:
                            st.markdown("<h3 class='glow-text'>📝 DETAILED REPORT</h3>", unsafe_allow_html=True)
                            for i, det in enumerate(detections, 1):
                                st.markdown(f"""
                                <div style='background: rgba(0, 212, 255, 0.1); 
                                            padding: 10px; margin: 5px 0; 
                                            border-left: 4px solid #00d4ff; 
                                            border-radius: 5px;'>
                                    <b>{i}. {det['emoji']} {det['class']}</b><br>
                                    Confidence: {det['confidence']:.2%}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Chart
                            st.markdown("<h3 class='glow-text'>📈 DISTRIBUTION CHART</h3>", unsafe_allow_html=True)
                            chart = create_detection_chart(detections)
                            if chart:
                                st.plotly_chart(chart, use_container_width=True)
                        else:
                            st.info("✅ No issues detected in this image!")
                    else:
                        st.error("❌ Failed to load model!")
        else:
            st.info("👆 Upload an image to start detection")
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()