import streamlit as st
import sqlite3
import os
import cv2
import numpy as np
import threading
import math
import time
import av
import winsound
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, AudioProcessorBase, WebRtcMode
from datetime import datetime
from PIL import Image
from ultralytics import YOLO


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Sentinel AI",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# DIRECTORIES
# =========================================================

DB_FILE = "events.db"
OUTPUT_DIR = "outputs"
VIDEO_DIR = "videos"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


model = load_model()


# =========================================================
# DATABASE
# =========================================================

def get_connection():
    return sqlite3.connect(DB_FILE)


def create_database():

    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            object_type TEXT,
            track_id INTEGER,
            evidence TEXT
        )
    """)

    conn.commit()
    conn.close()


create_database()


# =========================================================
# OBJECT CLASSES
# =========================================================

class_names = {
    0: "PERSON",
    2: "CAR",
    3: "MOTORCYCLE",
    5: "BUS",
    7: "TRUCK"
}

allowed_classes = [0, 2, 3, 5, 7]


# =========================================================
# RESTRICTED ZONE
# =========================================================

zone_x1, zone_y1 = 100, 100
zone_x2, zone_y2 = 500, 400


# =========================================================
# SAVE EVENT
# =========================================================

def save_event(object_type, track_id, evidence_path):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    conn = get_connection()

    conn.execute(
        """
        INSERT INTO events
        (timestamp, object_type, track_id, evidence)
        VALUES (?, ?, ?, ?)
        """,
        (
            timestamp,
            object_type,
            int(track_id),
            evidence_path
        )
    )

    conn.commit()
    conn.close()


# =========================================================
# CHECK RESTRICTED ZONE
# =========================================================

def inside_zone(center_x, center_y):

    return (
        zone_x1 <= center_x <= zone_x2
        and
        zone_y1 <= center_y <= zone_y2
    )


# =========================================================
# PHOTO ANALYSIS
# =========================================================

def analyze_photo(image_bytes):

    image = Image.open(image_bytes).convert("RGB")

    frame = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    results = model.predict(
        frame,
        classes=allowed_classes,
        conf=0.50,
        verbose=False
    )

    detections = []
    intrusion = False

    # Restricted zone

    cv2.rectangle(
        frame,
        (zone_x1, zone_y1),
        (zone_x2, zone_y2),
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        "RESTRICTED ZONE",
        (zone_x1 + 10, zone_y1 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    next_id = 1

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            object_name = class_names.get(
                class_id,
                "OBJECT"
            )

            track_id = next_id
            next_id += 1

            in_zone = inside_zone(
                center_x,
                center_y
            )

            if in_zone:
                intrusion = True

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = (
                f"{object_name} #{track_id} "
                f"{confidence * 100:.1f}%"
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(y1 - 10, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            detections.append({
                "object": object_name,
                "id": track_id,
                "confidence": confidence,
                "intrusion": in_zone
            })

    if intrusion:

        cv2.putText(
            frame,
            "INTRUSION ALERT!",
            (120, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )

    else:

        cv2.putText(
            frame,
            "STATUS: SAFE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    return frame, detections, intrusion


# =========================================================
# STYLING
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1220;
    color: white;
}

.title {
    font-size: 42px;
    font-weight: 800;
}

.subtitle {
    color: #9ca3af;
    font-size: 18px;
}

.alert {
    background-color: #7f1d1d;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}

.safe {
    background-color: #065f46;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🛡️ SENTINEL AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Intelligent Video Analytics Platform for Border Surveillance'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🛡️ Sentinel AI")

st.sidebar.markdown(
    "### Surveillance Modules"
)

st.sidebar.write("🧠 AI Detection")
st.sidebar.write("🆔 Object Tracking")
st.sidebar.write("🔴 Restricted Zone")
st.sidebar.write("🚨 Intrusion Alert")
st.sidebar.write("📸 Evidence Capture")
st.sidebar.write("📷 Camera Analysis")
st.sidebar.write("🎥 Video Analysis")
st.sidebar.write("💾 Event Database")

st.sidebar.divider()

st.sidebar.success("🟢 AI ENGINE ONLINE")
st.sidebar.success("🟢 DATABASE ONLINE")


# =========================================================
# SECURITY OVERVIEW
# =========================================================

conn = get_connection()

total_events = conn.execute(
    "SELECT COUNT(*) FROM events"
).fetchone()[0]

person_events = conn.execute(
    """
    SELECT COUNT(*)
    FROM events
    WHERE object_type='PERSON'
    """
).fetchone()[0]

vehicle_events = conn.execute(
    """
    SELECT COUNT(*)
    FROM events
    WHERE object_type IN
    ('CAR', 'MOTORCYCLE', 'BUS', 'TRUCK')
    """
).fetchone()[0]

unique_objects = conn.execute(
    """
    SELECT COUNT(DISTINCT track_id)
    FROM events
    """
).fetchone()[0]

conn.close()


st.subheader("📊 Security Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "🚨 Total Alerts",
        total_events
    )

with col2:

    st.metric(
        "👤 Person Alerts",
        person_events
    )

with col3:

    st.metric(
        "🚗 Vehicle Alerts",
        vehicle_events
    )

with col4:

    st.metric(
        "🆔 Tracked Objects",
        unique_objects
    )


st.divider()


# =========================================================
# CAMERA PHOTO ANALYSIS
# =========================================================

st.subheader("📷 Camera Photo Analysis")

st.info(
    "Take a photo and BorderGuard AI will automatically "
    "analyze it for people and vehicles."
)

camera_image = st.camera_input(
    "📸 Take Photo"
)


if camera_image is not None:

    st.write("### 🧠 AI Analysis")

    annotated_frame, detections, intrusion = analyze_photo(
        camera_image
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    original_path = os.path.join(
        OUTPUT_DIR,
        f"camera_{timestamp}.jpg"
    )

    with open(
        original_path,
        "wb"
    ) as file:

        file.write(
            camera_image.getbuffer()
        )

    evidence_path = os.path.join(
        OUTPUT_DIR,
        f"camera_evidence_{timestamp}.jpg"
    )

    cv2.imwrite(
        evidence_path,
        annotated_frame
    )

    result_rgb = cv2.cvtColor(
        annotated_frame,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        result_rgb,
        caption="🧠 AI Detection Result",
        width="stretch"
    )

    if detections:

        st.success(
            f"🧠 {len(detections)} object(s) detected."
        )

        for detection in detections:

            object_name = detection["object"]
            track_id = detection["id"]
            confidence = detection["confidence"]
            in_zone = detection["intrusion"]

            if object_name == "PERSON":
                icon = "🧍"

            elif object_name == "BUS":
                icon = "🚌"

            elif object_name == "CAR":
                icon = "🚗"

            elif object_name == "MOTORCYCLE":
                icon = "🏍️"

            elif object_name == "TRUCK":
                icon = "🚚"

            else:
                icon = "📦"

            st.write(
                f"{icon} **{object_name} #{track_id}**"
            )

            st.write(
                f"Confidence: "
                f"**{confidence * 100:.1f}%**"
            )

            if in_zone:

                st.error(
                    f"🚨 INTRUSION ALERT! "
                    f"{object_name} #{track_id} "
                    f"entered the restricted zone!"
                )

                save_event(
                    object_name,
                    track_id,
                    evidence_path
                )

                st.write(
                    "📸 Evidence saved"
                )

                st.write(
                    "💾 Event saved to database"
                )

            else:

                st.success(
                    f"🟢 {object_name} detected "
                    f"outside restricted zone."
                )

    else:

        st.info(
            "No person or supported vehicle detected."
        )


st.divider()


# =========================================================
# VIDEO AI ANALYSIS
# =========================================================

st.subheader("🎥 Video AI Analysis")

st.info(
    "Upload a surveillance video. BorderGuard AI will "
    "analyze the video frame-by-frame and count "
    "people and vehicles."
)

uploaded_video = st.file_uploader(
    "🎥 Upload Surveillance Video",
    type=["mp4", "avi", "mov"]
)


if uploaded_video is not None:

    input_path = os.path.join(
        OUTPUT_DIR,
        "uploaded_input_video.mp4"
    )

    with open(
        input_path,
        "wb"
    ) as file:

        file.write(
            uploaded_video.getbuffer()
        )

    st.success(
        "🎥 Video uploaded successfully."
    )

    st.video(uploaded_video)

    analyze_video = st.button(
        "🧠 START AI VIDEO ANALYSIS"
    )


    if analyze_video:

        cap = cv2.VideoCapture(
            input_path
        )

        if not cap.isOpened():

            st.error(
                "❌ Could not open the uploaded video."
            )

        else:

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            if fps <= 0:
                fps = 25

            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            if width <= 0 or height <= 0:

                cap.release()

                st.error(
                    "❌ Invalid video dimensions."
                )

                st.stop()


            # =================================================
            # OUTPUT VIDEO
            # =================================================

            output_timestamp = datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S"
            )

            output_video_path = os.path.join(
                OUTPUT_DIR,
                f"analyzed_{output_timestamp}.mp4"
            )

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                output_video_path,
                fourcc,
                fps,
                (width, height)
            )

            if not writer.isOpened():

                cap.release()

                st.error(
                    "❌ Could not create output video."
                )

                st.stop()


            # =================================================
            # PROGRESS
            # =================================================

            progress = st.progress(0)

            status_box = st.empty()

            frame_info = st.empty()


            # =================================================
            # STABLE OBJECT TRACKING
            #
            # THIS IS THE IMPORTANT FIX
            # =================================================

            stable_tracks = {

                "PERSON": [],
                "CAR": [],
                "MOTORCYCLE": [],
                "BUS": [],
                "TRUCK": []

            }


            next_stable_id = {

                "PERSON": 1,
                "CAR": 1,
                "MOTORCYCLE": 1,
                "BUS": 1,
                "TRUCK": 1

            }


            # How many frames an object can disappear
            # before we consider it gone.

            max_missing_frames = 20


            # Distance used to match the same object
            # between consecutive frames.

            match_distance = max(
                50,
                int(min(width, height) * 0.08)
            )


            # =================================================
            # FINAL OBJECT IDs
            # =================================================

            people_ids = set()
            car_ids = set()
            motorcycle_ids = set()
            bus_ids = set()
            truck_ids = set()


            # =================================================
            # EVENTS
            # =================================================

            event_count = 0

            seen_events = set()

            frame_number = 0


            # =================================================
            # PROCESS VIDEO
            # =================================================

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frame_number += 1


                # =================================================
                # YOLO TRACKING
                # =================================================

                results = model.track(
                    frame,
                    classes=allowed_classes,
                    conf=0.20,
                    tracker="bytetrack.yaml",
                    persist=True,
                    verbose=False
                )


                intrusion_this_frame = False


                # =================================================
                # DETECTIONS FOR THIS FRAME
                # =================================================

                current_detections = {

                    "PERSON": [],
                    "CAR": [],
                    "MOTORCYCLE": [],
                    "BUS": [],
                    "TRUCK": []

                }


                for result in results:

                    if result.boxes is None:
                        continue


                    boxes = (
                        result.boxes.xyxy
                        .cpu()
                        .numpy()
                    )

                    classes = (
                        result.boxes.cls
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                    confidences = (
                        result.boxes.conf
                        .cpu()
                        .numpy()
                    )


                    for box, class_id, confidence in zip(
                        boxes,
                        classes,
                        confidences
                    ):

                        object_name = class_names.get(
                            class_id,
                            "OBJECT"
                        )

                        if object_name not in stable_tracks:
                            continue


                        x1, y1, x2, y2 = map(
                            int,
                            box
                        )

                        center_x = (
                            x1 + x2
                        ) // 2

                        center_y = (
                            y1 + y2
                        ) // 2


                        current_detections[
                            object_name
                        ].append({

                            "box": (
                                x1,
                                y1,
                                x2,
                                y2
                            ),

                            "center": (
                                center_x,
                                center_y
                            ),

                            "confidence": float(
                                confidence
                            )

                        })


                # =================================================
                # STABLE ID MATCHING
                # =================================================

                assigned_objects = []


                for object_name in current_detections:

                    detections_for_class = (
                        current_detections[
                            object_name
                        ]
                    )


                    tracks_for_class = (
                        stable_tracks[
                            object_name
                        ]
                    )


                    # Mark old tracks as missing

                    for track in tracks_for_class:

                        track["missing"] += 1


                    used_track_ids = set()


                    for detection in detections_for_class:

                        cx, cy = detection[
                            "center"
                        ]


                        best_track = None

                        best_distance = float(
                            "inf"
                        )


                        # Find nearest existing object

                        for track in tracks_for_class:

                            if track["stable_id"] in used_track_ids:
                                continue


                            if (
                                track["missing"]
                                > max_missing_frames
                            ):
                                continue


                            old_cx, old_cy = (
                                track["center"]
                            )


                            distance = (
                                (cx - old_cx) ** 2
                                +
                                (cy - old_cy) ** 2
                            ) ** 0.5


                            if (
                                distance
                                < best_distance
                                and
                                distance
                                <= match_distance
                            ):

                                best_distance = distance

                                best_track = track


                        # =========================================
                        # EXISTING OBJECT
                        # =========================================

                        if best_track is not None:

                            best_track[
                                "center"
                            ] = (
                                cx,
                                cy
                            )

                            best_track[
                                "box"
                            ] = detection[
                                "box"
                            ]

                            best_track[
                                "confidence"
                            ] = detection[
                                "confidence"
                            ]

                            best_track[
                                "missing"
                            ] = 0


                            stable_id = best_track[
                                "stable_id"
                            ]


                        # =========================================
                        # NEW OBJECT
                        # =========================================

                        else:

                            stable_id = (
                                next_stable_id[
                                    object_name
                                ]
                            )

                            next_stable_id[
                                object_name
                            ] += 1


                            new_track = {

                                "stable_id": stable_id,

                                "center": (
                                    cx,
                                    cy
                                ),

                                "box": detection[
                                    "box"
                                ],

                                "confidence": detection[
                                    "confidence"
                                ],

                                "missing": 0

                            }


                            tracks_for_class.append(
                                new_track
                            )


                            best_track = new_track


                        used_track_ids.add(
                            stable_id
                        )


                        assigned_objects.append({

                            "object": object_name,

                            "stable_id": stable_id,

                            "box": detection[
                                "box"
                            ],

                            "center": (
                                cx,
                                cy
                            ),

                            "confidence": detection[
                                "confidence"
                            ]

                        })


                    # Remove very old tracks

                    stable_tracks[
                        object_name
                    ] = [

                        track

                        for track in tracks_for_class

                        if track["missing"]
                        <= max_missing_frames

                    ]


                # =================================================
                # DRAW OBJECTS
                # =================================================

                for obj in assigned_objects:

                    object_name = obj[
                        "object"
                    ]

                    stable_id = obj[
                        "stable_id"
                    ]

                    x1, y1, x2, y2 = obj[
                        "box"
                    ]

                    center_x, center_y = obj[
                        "center"
                    ]

                    confidence = obj[
                        "confidence"
                    ]


                    # Add ID to final sets

                    if object_name == "PERSON":

                        people_ids.add(
                            stable_id
                        )

                    elif object_name == "CAR":

                        car_ids.add(
                            stable_id
                        )

                    elif object_name == "MOTORCYCLE":

                        motorcycle_ids.add(
                            stable_id
                        )

                    elif object_name == "BUS":

                        bus_ids.add(
                            stable_id
                        )

                    elif object_name == "TRUCK":

                        truck_ids.add(
                            stable_id
                        )


                    # Check restricted zone

                    in_zone = inside_zone(
                        center_x,
                        center_y
                    )


                    # =================================================
                    # BOX
                    # =================================================

                    if in_zone:

                        box_color = (
                            0,
                            0,
                            255
                        )

                    else:

                        box_color = (
                            0,
                            255,
                            0
                        )


                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        2
                    )


                    # =================================================
                    # LABEL
                    # =================================================

                    label = (
                        f"{object_name} "
                        f"#{stable_id} "
                        f"{confidence * 100:.0f}%"
                    )


                    cv2.putText(
                        frame,
                        label,
                        (
                            x1,
                            max(
                                y1 - 10,
                                20
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_color,
                        2
                    )


                    # =================================================
                    # INTRUSION
                    # =================================================

                    if in_zone:

                        intrusion_this_frame = True


                        cv2.putText(
                            frame,
                            "INTRUSION ALERT!",
                            (120, 55),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            3
                        )


                        event_key = (

                            object_name,

                            int(stable_id)

                        )


                        if event_key not in seen_events:

                            seen_events.add(
                                event_key
                            )


                            timestamp = datetime.now().strftime(
                                "%Y-%m-%d_%H-%M-%S"
                            )


                            evidence_path = os.path.join(
                                OUTPUT_DIR,
                                f"video_intrusion_"
                                f"{timestamp}_"
                                f"{object_name}_"
                                f"{stable_id}.jpg"
                            )


                            cv2.imwrite(
                                evidence_path,
                                frame
                            )


                            save_event(
                                object_name,
                                stable_id,
                                evidence_path
                            )


                            event_count += 1


                # =================================================
                # RESTRICTED ZONE
                # =================================================

                cv2.rectangle(
                    frame,
                    (zone_x1, zone_y1),
                    (zone_x2, zone_y2),
                    (0, 0, 255),
                    2
                )


                cv2.putText(
                    frame,
                    "RESTRICTED ZONE",
                    (
                        zone_x1 + 10,
                        zone_y1 + 30
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )


                # =================================================
                # STATUS
                # =================================================

                if intrusion_this_frame:

                    cv2.putText(
                        frame,
                        "STATUS: INTRUSION",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )


                    status_box.error(
                        f"🚨 INTRUSION DETECTED "
                        f"| Frame {frame_number}"
                    )

                else:

                    cv2.putText(
                        frame,
                        "STATUS: SAFE",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )


                # =================================================
                # WRITE FRAME
                # =================================================

                writer.write(
                    frame
                )


                # =================================================
                # PROGRESS
                # =================================================

                if total_frames > 0:

                    progress_value = (
                        frame_number
                        /
                        total_frames
                    )


                    progress.progress(
                        min(
                            progress_value,
                            1.0
                        )
                    )


                frame_info.write(
                    f"Processing frame "
                    f"{frame_number} / "
                    f"{total_frames}"
                )


            # =================================================
            # RELEASE VIDEO
            # =================================================

            cap.release()

            writer.release()

            progress.progress(1.0)


            status_box.success(
                "✅ Video analysis completed."
            )

            frame_info.empty()


            # =================================================
            # FINAL COUNTS
            # =================================================

            people_count = len(
                people_ids
            )

            car_count = len(
                car_ids
            )

            motorcycle_count = len(
                motorcycle_ids
            )

            bus_count = len(
                bus_ids
            )

            truck_count = len(
                truck_ids
            )


            total_detected = (
                people_count
                +
                car_count
                +
                motorcycle_count
                +
                bus_count
                +
                truck_count
            )


            # =================================================
            # AI DETECTION SUMMARY
            # =================================================

            st.subheader(
                "🧠 AI Detection Summary"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "🧍 People",
                    people_count
                )

                st.metric(
                    "🚌 Buses",
                    bus_count
                )


            with col2:

                st.metric(
                    "🚗 Cars",
                    car_count
                )

                st.metric(
                    "🏍️ Motorcycles",
                    motorcycle_count
                )


            with col3:

                st.metric(
                    "🚚 Trucks",
                    truck_count
                )

                st.metric(
                    "📦 Total Objects",
                    total_detected
                )


            st.divider()


            # =================================================
            # DETECTED OBJECTS
            # =================================================

            st.subheader(
                "📋 Detected Objects"
            )


            if people_count > 0:

                for person_id in sorted(
                    people_ids
                ):

                    st.write(
                        f"🧍 **Person #{person_id}**"
                    )


            if bus_count > 0:

                for bus_id in sorted(
                    bus_ids
                ):

                    st.write(
                        f"🚌 **Bus #{bus_id}**"
                    )


            if car_count > 0:

                for car_id in sorted(
                    car_ids
                ):

                    st.write(
                        f"🚗 **Car #{car_id}**"
                    )


            if motorcycle_count > 0:

                for motorcycle_id in sorted(
                    motorcycle_ids
                ):

                    st.write(
                        f"🏍️ **Motorcycle #{motorcycle_id}**"
                    )


            if truck_count > 0:

                for truck_id in sorted(
                    truck_ids
                ):

                    st.write(
                        f"🚚 **Truck #{truck_id}**"
                    )


            if total_detected == 0:

                st.warning(
                    "⚠️ No supported objects were detected."
                )


            st.divider()


            # =================================================
            # INTRUSION RESULT
            # =================================================

            if event_count > 0:

                st.error(
                    f"🚨 **Intrusion events detected: "
                    f"{event_count}**"
                )

            else:

                st.success(
                    "🟢 **Intrusion events detected: 0**"
                )


            # =================================================
            # ANALYZED VIDEO
            # =================================================

            if os.path.exists(
                output_video_path
            ):

                st.subheader(
                    "🎬 AI Analyzed Video"
                )

                st.video(
                    output_video_path
                )

                st.success(
                    "✅ Analyzed video saved successfully."
                )

                st.caption(
                    f"📁 {output_video_path}"
                )



st.divider()


# =========================================================
# LIVE CAMERA RECORDING
# =========================================================

st.subheader("🎥 Live Camera Recording")

st.info(
    "Start the live camera to record video while BorderGuard AI "
    "detects people and vehicles in real time."
)

LIVE_RECORDING_DIR = VIDEO_DIR
os.makedirs(LIVE_RECORDING_DIR, exist_ok=True)

live_lock = threading.Lock()

if "live_recording_path" not in st.session_state:
    st.session_state.live_recording_path = ""

if "live_recording_ready" not in st.session_state:
    st.session_state.live_recording_ready = False

if "alarm_silenced" not in st.session_state:
    st.session_state.alarm_silenced = False

if "alarm_command" not in st.session_state:
    st.session_state.alarm_command = False

if "live_alarm_status" not in st.session_state:
    st.session_state.live_alarm_status = False

if "alarm_shared_state" not in st.session_state:
    st.session_state.alarm_shared_state = {
        "silenced": False,
        "intrusion": False,
        "lock": threading.Lock()
    }

alarm_shared_state = st.session_state.alarm_shared_state

alarm_col1, alarm_col2 = st.columns(2)

with alarm_col1:
    if st.button("🛑 STOP ALARM", key="stop_live_alarm"):
        with alarm_shared_state["lock"]:
            alarm_shared_state["silenced"] = True
        st.session_state.alarm_silenced = True

with alarm_col2:
    if st.button("🔔 RESET ALARM", key="reset_live_alarm"):
        with alarm_shared_state["lock"]:
            if not alarm_shared_state["intrusion"]:
                alarm_shared_state["silenced"] = False
                st.session_state.alarm_silenced = False
                st.success("Alarm reset. The zone is safe.")
            else:
                st.warning(
                    "The restricted zone is occupied. "
                    "The alarm cannot be reset until the zone is safe."
                )
last_intrusion = False


class LiveVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.writer = None
        self.output_path = ""
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 20
        self.frame_count = 0
        self.event_count = 0
        self.seen_events = set()
        self.alarm_active = False
        self.alarm_thread = None
        self.alarm_stop_event = threading.Event()
        self.previous_intrusion = False

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.writer is None:

            self.frame_height, self.frame_width = img.shape[:2]

            timestamp = datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S"
            )

            self.output_path = os.path.join(
                LIVE_RECORDING_DIR,
                f"live_recording_{timestamp}.mp4"
            )

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height)
            )

            with live_lock:
                st.session_state.live_recording_path = self.output_path
                st.session_state.live_recording_ready = False

        results = model.track(
            img,
            classes=allowed_classes,
            conf=0.20,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )

        intrusion = False

        for result in results:

            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = [-1] * len(boxes)

            for box, class_id, confidence, track_id in zip(
                boxes, classes, confidences, track_ids
            ):

                object_name = class_names.get(class_id, "OBJECT")

                x1, y1, x2, y2 = map(int, box)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                in_zone = inside_zone(center_x, center_y)

                if in_zone:
                    intrusion = True

                box_color = (0, 0, 255) if in_zone else (0, 255, 0)

                cv2.rectangle(
                    img, (x1, y1), (x2, y2), box_color, 2
                )

                if track_id != -1:
                    label = (
                        f"{object_name} #{int(track_id)} "
                        f"{confidence * 100:.0f}%"
                    )
                else:
                    label = (
                        f"{object_name} {confidence * 100:.0f}%"
                    )

                cv2.putText(
                    img,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2
                )

                if in_zone and track_id != -1:

                    event_key = (object_name, int(track_id))

                    if event_key not in self.seen_events:

                        self.seen_events.add(event_key)

                        timestamp = datetime.now().strftime(
                            "%Y-%m-%d_%H-%M-%S"
                        )

                        evidence_path = os.path.join(
                            OUTPUT_DIR,
                            f"live_intrusion_{timestamp}_"
                            f"{object_name}_{int(track_id)}.jpg"
                        )

                        cv2.imwrite(evidence_path, img)

                        save_event(
                            object_name,
                            int(track_id),
                            evidence_path
                        )

                        self.event_count += 1

        # -----------------------------------------------------
        # ALARM CONTROL
        # Stop button silences the alarm until the zone is safe.
        # -----------------------------------------------------

        with alarm_shared_state["lock"]:

            alarm_shared_state["intrusion"] = intrusion

            if not intrusion:
                alarm_shared_state["silenced"] = False

            should_sound = (
                intrusion
                and not alarm_shared_state["silenced"]
            )

        if should_sound and not self.alarm_active:

            self.alarm_active = True
            self.alarm_stop_event.clear()

            def alarm():
                try:
                    while not self.alarm_stop_event.is_set():
                        with alarm_shared_state["lock"]:
                            if alarm_shared_state["silenced"]:
                                break

                        winsound.Beep(1500, 1000)

                finally:
                    self.alarm_active = False

            self.alarm_thread = threading.Thread(
                target=alarm,
                daemon=True
            )
            self.alarm_thread.start()

        elif not should_sound:
            self.alarm_stop_event.set()

        # -----------------------------------------------------
        # DRAW RESTRICTED ZONE
        # -----------------------------------------------------

        cv2.rectangle(
            img,
            (zone_x1, zone_y1),
            (zone_x2, zone_y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            img,
            "RESTRICTED ZONE",
            (zone_x1 + 10, zone_y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        if intrusion:

            cv2.putText(
                img,
                "INTRUSION ALERT!",
                (120, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

            with alarm_shared_state["lock"]:
                silenced = alarm_shared_state["silenced"]

            if silenced:
                status_text = "ALARM SILENCED"
            else:
                status_text = "ALARM ACTIVE"

            cv2.putText(
                img,
                status_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255) if silenced else (0, 0, 255),
                2
            )

        else:

            cv2.putText(
                img,
                "STATUS: SAFE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        if self.writer is not None:
            self.writer.write(img)

        self.previous_intrusion = intrusion

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

    def on_ended(self):

        self.alarm_stop_event.set()

        if self.writer is not None:
            self.writer.release()
            self.writer = None

            with live_lock:
                if self.output_path:
                    st.session_state.live_recording_path = self.output_path
                    st.session_state.live_recording_ready = (
                        os.path.exists(self.output_path)
                        and os.path.getsize(self.output_path) > 0
                    )


live_ctx = webrtc_streamer(
    key="borderguard-live-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=LiveVideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {
                "urls": ["stun:stun.l.google.com:19302"]
            }
        ]
    }
)

if live_ctx.state.playing:

    st.success("🔴 LIVE RECORDING ACTIVE")

    with alarm_shared_state["lock"]:
        alarm_is_silenced = alarm_shared_state["silenced"]
        current_intrusion = alarm_shared_state["intrusion"]

    if current_intrusion and alarm_is_silenced:
        st.warning("🔇 Alarm silenced by officer. Intrusion is still present.")
    elif current_intrusion:
        st.error("🚨 Intrusion detected. Alarm is active.")
    else:
        st.success("🟢 Restricted zone is safe.")

else:

    if st.session_state.live_recording_ready:
        st.success("⏹️ Recording stopped successfully.")
    else:
        st.info("Click START above to begin live recording.")

live_path = st.session_state.live_recording_path

if (
    live_path
    and os.path.exists(live_path)
    and os.path.getsize(live_path) > 0
):

    st.write("### 🎬 Recorded Live Video")
    st.video(live_path)
    st.caption(f"📁 {live_path}")

    analyze_live_recording = st.button(
        "🧠 ANALYZE RECORDED LIVE VIDEO",
        key="analyze_live_recording"
    )

    if analyze_live_recording:

        cap = cv2.VideoCapture(live_path)

        if not cap.isOpened():
            st.error("❌ Could not open the recorded video.")

        else:

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 20

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            analyzed_live_path = os.path.join(
                OUTPUT_DIR,
                f"analyzed_live_{timestamp}.mp4"
            )

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(
                analyzed_live_path,
                fourcc,
                fps,
                (width, height)
            )

            progress = st.progress(0)
            status_box = st.empty()
            frame_info = st.empty()

            people_ids = set()
            car_ids = set()
            motorcycle_ids = set()
            bus_ids = set()
            truck_ids = set()

            event_count = 0
            seen_events = set()
            frame_number = 0

            while True:

                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1

                results = model.track(
                    frame,
                    classes=allowed_classes,
                    conf=0.20,
                    tracker="bytetrack.yaml",
                    persist=True,
                    verbose=False
                )

                intrusion_this_frame = False

                for result in results:

                    if result.boxes is None:
                        continue

                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    confidences = result.boxes.conf.cpu().numpy()

                    if result.boxes.id is not None:
                        track_ids = result.boxes.id.cpu().numpy().astype(int)
                    else:
                        track_ids = [-1] * len(boxes)

                    for box, class_id, confidence, track_id in zip(
                        boxes, classes, confidences, track_ids
                    ):

                        object_name = class_names.get(class_id, "OBJECT")

                        x1, y1, x2, y2 = map(int, box)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        in_zone = inside_zone(center_x, center_y)

                        if object_name == "PERSON" and track_id != -1:
                            people_ids.add(int(track_id))
                        elif object_name == "CAR" and track_id != -1:
                            car_ids.add(int(track_id))
                        elif object_name == "MOTORCYCLE" and track_id != -1:
                            motorcycle_ids.add(int(track_id))
                        elif object_name == "BUS" and track_id != -1:
                            bus_ids.add(int(track_id))
                        elif object_name == "TRUCK" and track_id != -1:
                            truck_ids.add(int(track_id))

                        if in_zone:
                            intrusion_this_frame = True

                            if track_id != -1:
                                event_key = (object_name, int(track_id))

                                if event_key not in seen_events:
                                    seen_events.add(event_key)

                                    timestamp = datetime.now().strftime(
                                        "%Y-%m-%d_%H-%M-%S"
                                    )

                                    evidence_path = os.path.join(
                                        OUTPUT_DIR,
                                        f"analyzed_live_intrusion_{timestamp}_"
                                        f"{object_name}_{int(track_id)}.jpg"
                                    )

                                    cv2.imwrite(evidence_path, frame)
                                    save_event(object_name, int(track_id), evidence_path)
                                    event_count += 1

                        color = (0, 0, 255) if in_zone else (0, 255, 0)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        if track_id != -1:
                            label = (
                                f"{object_name} #{int(track_id)} "
                                f"{confidence * 100:.0f}%"
                            )
                        else:
                            label = f"{object_name} {confidence * 100:.0f}%"

                        cv2.putText(
                            frame,
                            label,
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2
                        )

                cv2.rectangle(
                    frame,
                    (zone_x1, zone_y1),
                    (zone_x2, zone_y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "RESTRICTED ZONE",
                    (zone_x1 + 10, zone_y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                if intrusion_this_frame:
                    cv2.putText(
                        frame,
                        "INTRUSION ALERT!",
                        (120, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3
                    )
                    status_box.error(
                        f"🚨 INTRUSION DETECTED | Frame {frame_number}"
                    )
                else:
                    cv2.putText(
                        frame,
                        "STATUS: SAFE",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                writer.write(frame)

                if total_frames > 0:
                    progress.progress(min(frame_number / total_frames, 1.0))

                frame_info.write(
                    f"Processing frame {frame_number} / {total_frames}"
                )

            cap.release()
            writer.release()
            progress.progress(1.0)

            status_box.success(
                "✅ Recorded live video analysis completed."
            )
            frame_info.empty()

            people_count = len(people_ids)
            car_count = len(car_ids)
            motorcycle_count = len(motorcycle_ids)
            bus_count = len(bus_ids)
            truck_count = len(truck_ids)

            total_detected = (
                people_count + car_count + motorcycle_count
                + bus_count + truck_count
            )

            st.subheader("🧠 Live Recording AI Detection Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("🧍 People", people_count)
                st.metric("🚌 Buses", bus_count)

            with col2:
                st.metric("🚗 Cars", car_count)
                st.metric("🏍️ Motorcycles", motorcycle_count)

            with col3:
                st.metric("🚚 Trucks", truck_count)
                st.metric("📦 Total Objects", total_detected)

            st.divider()
            st.subheader("📋 Detected Objects")

            for person_id in sorted(people_ids):
                st.write(f"🧍 **Person #{person_id}**")

            for bus_id in sorted(bus_ids):
                st.write(f"🚌 **Bus #{bus_id}**")

            for car_id in sorted(car_ids):
                st.write(f"🚗 **Car #{car_id}**")

            for motorcycle_id in sorted(motorcycle_ids):
                st.write(f"🏍️ **Motorcycle #{motorcycle_id}**")

            for truck_id in sorted(truck_ids):
                st.write(f"🚚 **Truck #{truck_id}**")

            if total_detected == 0:
                st.warning("⚠️ No supported objects were detected.")

            if event_count > 0:
                st.error(f"🚨 **Intrusion events detected: {event_count}**")
            else:
                st.success("🟢 **Intrusion events detected: 0**")

            st.subheader("🎬 AI Analyzed Live Recording")
            st.video(analyzed_live_path)
            st.success("✅ Analyzed live recording saved successfully.")
            st.caption(f"📁 {analyzed_live_path}")

st.divider()



# =========================================================
# SECURITY STATUS
# =========================================================

st.subheader(
    "🚦 Security Status"
)


if total_events > 0:

    st.markdown(
        """
        <div class="alert">
        🚨 INTRUSION EVENTS DETECTED
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <div class="safe">
        🟢 BORDER STATUS: SAFE
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# =========================================================
# EVENT HISTORY
# =========================================================

with st.expander(
    "📝 Intrusion Event History",
    expanded=False
):

    conn = get_connection()

    events = conn.execute(
        """
        SELECT
            id,
            timestamp,
            object_type,
            track_id,
            evidence
        FROM events
        ORDER BY id DESC
        """
    ).fetchall()

    conn.close()

    if events:

        for event in events:

            event_id = event[0]
            timestamp = event[1]
            object_type = event[2]
            track_id = event[3]
            evidence = event[4]

            with st.expander(
                f"🚨 Event #{event_id} | "
                f"{object_type} #{track_id} | "
                f"{timestamp}"
            ):

                st.write(
                    f"**Object:** {object_type}"
                )

                st.write(
                    f"**Tracking ID:** #{track_id}"
                )

                st.write(
                    f"**Time:** {timestamp}"
                )

                if (
                    evidence
                    and
                    os.path.exists(evidence)
                ):

                    st.image(
                        evidence,
                        caption="📸 Evidence",
                        width="stretch"
                    )

                else:

                    st.warning(
                        "Evidence image not found."
                    )

    else:

        st.success(
            "🟢 No intrusion events recorded."
        )


st.divider()


# =========================================================
# EVIDENCE VIEWER
# =========================================================

with st.expander(
    "📸 Evidence Viewer",
    expanded=False
):

    evidence_files = []

    if os.path.exists(
        OUTPUT_DIR
    ):

        evidence_files = [

            f

            for f in os.listdir(
                OUTPUT_DIR
            )

            if f.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            )

        ]

    if evidence_files:

        selected_file = st.selectbox(
            "Select evidence image",
            sorted(
                evidence_files,
                reverse=True
            )
        )

        selected_path = os.path.join(
            OUTPUT_DIR,
            selected_file
        )

        st.image(
            selected_path,
            caption=selected_file,
            width="stretch"
        )

    else:

        st.info(
            "No evidence images available."
        )


st.divider()




# =========================================================
# AI PIPELINE
# =========================================================

st.subheader(
    "🏗️ AI Surveillance Pipeline"
)


st.markdown("""
**📹 CCTV / Camera**
→
**🧠 YOLO Detection**
→
**🆔 ByteTrack**
→
**🔴 Restricted Zone**
→
**🚨 Intrusion Alert**
→
**📸 Evidence**
→
**💾 Database**
""")


st.divider()


# =========================================================
# FOOTER
# =========================================================

st.caption(
    "BorderGuard AI | Smart India Hackathon Prototype | "
    "AI + Computer Vision + Object Tracking"
)
