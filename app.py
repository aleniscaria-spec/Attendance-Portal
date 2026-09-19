import os
import re
import csv
import io
import base64
import logging
from datetime import datetime
from functools import wraps

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    session,
    flash,
)
import numpy as np
from PIL import Image

from face_engine import FaceRecognitionEngine

# Configure application logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("attendance_app")

# Initialize Flask App
app = Flask(__name__)

# Security configuration via environment variables
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "flask-attendance-secret-key-2026")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "password123")

# Storage paths
DATASET_DIR = "dataset"
ATTENDANCE_FILE = "attendance.csv"

# Initialize Face Recognition Engine
engine = FaceRecognitionEngine(dataset_dir=DATASET_DIR, tolerance=0.50)


# ==================================================
# HELPER FUNCTIONS
# ==================================================


def init_attendance_csv():
    """Ensure attendance.csv exists with correct header."""
    if not os.path.exists(ATTENDANCE_FILE):
        try:
            with open(ATTENDANCE_FILE, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Date", "Time"])
            logger.info(f"Created new '{ATTENDANCE_FILE}'")
        except Exception as e:
            logger.error(f"Error initializing attendance.csv: {e}")


# Ensure CSV is present at startup
init_attendance_csv()


def admin_required(f):
    """Decorator to require admin authentication for protected views."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_logged_in"):
            flash("Administrator login required to access this page.", "warning")
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)

    return decorated_function


def decode_base64_image(base64_string):
    """
    Decode a base64 encoded image string (data URL or raw base64)
    into an RGB NumPy array.
    """
    if not base64_string:
        return None

    try:
        # Strip header prefix if present (e.g. data:image/jpeg;base64,...)
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        image_bytes = base64.b64decode(base64_string)
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB mode (removes alpha channel if PNG)
        pil_image = pil_image.convert("RGB")
        return np.array(pil_image)
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        return None


def sanitize_student_name(raw_name):
    """
    Sanitize student name for safe filesystem storage:
    - Trims whitespace
    - Replaces internal whitespace with underscores
    - Strips invalid characters to prevent directory traversal
    """
    cleaned = raw_name.strip()
    # Remove any dangerous characters (e.g. slashes, dots, null bytes)
    cleaned = re.sub(r"[^\w\s\-]", "", cleaned)
    # Replace multiple spaces with a single space
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def is_already_marked_today(name, target_date=None):
    """
    Check whether a student already has an attendance record for today.
    """
    if not os.path.exists(ATTENDANCE_FILE):
        return False

    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")

    target_name = name.strip().lower()

    try:
        with open(ATTENDANCE_FILE, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            # Skip header
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    row_name = row[0].strip().lower()
                    row_date = row[1].strip()
                    if row_name == target_name and row_date == target_date:
                        return True
    except Exception as e:
        logger.error(f"Error reading {ATTENDANCE_FILE}: {e}")

    return False


def record_attendance(name):
    """
    Mark student attendance:
    1. Check if student already marked today.
    2. If not, append new record.
    Returns: (marked: bool, message: str)
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Prevent duplicate attendance on the same day
    if is_already_marked_today(name, date_str):
        return False, "Already marked today"

    try:
        with open(ATTENDANCE_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([name, date_str, time_str])
        logger.info(f"Attendance recorded for {name} on {date_str} at {time_str}")
        return True, "Attendance marked"
    except Exception as e:
        logger.error(f"Failed to record attendance: {e}")
        return False, "Error saving attendance record"


def get_all_attendance_records():
    """Retrieve all attendance rows, newest first."""
    records = []
    if not os.path.exists(ATTENDANCE_FILE):
        return records

    try:
        with open(ATTENDANCE_FILE, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    records.append(
                        {
                            "name": row[0].strip(),
                            "date": row[1].strip(),
                            "time": row[2].strip(),
                        }
                    )
    except Exception as e:
        logger.error(f"Error reading attendance records: {e}")

    # Return newest records first
    records.reverse()
    return records


def get_dashboard_metrics():
    """Calculate counts for dashboard."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    records = get_all_attendance_records()

    today_count = sum(1 for r in records if r["date"] == today_str)
    total_count = len(records)
    student_count = engine.student_count()

    return {
        "registered_students": student_count,
        "today_attendance": today_count,
        "total_attendance": total_count,
    }


# ==================================================
# FLASK ROUTES
# ==================================================


@app.route("/")
def portal():
    """Main portal landing page."""
    metrics = get_dashboard_metrics()
    return render_template("portal.html", metrics=metrics)


@app.route("/user")
def student_portal():
    """Student attendance scanner page using browser camera."""
    return render_template("index.html")


@app.route("/recognize", methods=["POST"])
def recognize():
    """
    Process incoming browser camera frame:
    1. Receive base64 image.
    2. Decode image.
    3. Run face recognition engine.
    4. Mark attendance if recognized (only once per day).
    5. Return JSON result.
    """
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return (
            jsonify(
                {
                    "success": False,
                    "recognized": False,
                    "message": "No image received",
                }
            ),
            400,
        )

    base64_image = data.get("image", "")
    rgb_image = decode_base64_image(base64_image)

    if rgb_image is None:
        return (
            jsonify(
                {
                    "success": False,
                    "recognized": False,
                    "message": "Invalid image data",
                }
            ),
            400,
        )

    # Perform face recognition
    recognition_result = engine.recognize(rgb_image)
    status = recognition_result["status"]

    if status == "no_face":
        return jsonify(
            {
                "success": False,
                "recognized": False,
                "status": "no_face",
                "message": "No face detected",
            }
        )

    if status == "unknown":
        return jsonify(
            {
                "success": True,
                "recognized": False,
                "status": "unknown",
                "name": "Unknown",
                "distance": recognition_result.get("distance"),
                "attendance_marked": False,
                "message": "Unknown student",
            }
        )

    if status == "recognized":
        student_name = recognition_result["name"]
        distance = recognition_result.get("distance")

        # Mark attendance with duplicate prevention
        attendance_marked, attendance_msg = record_attendance(student_name)

        return jsonify(
            {
                "success": True,
                "recognized": True,
                "status": "recognized",
                "name": student_name,
                "distance": distance,
                "attendance_marked": attendance_marked,
                "message": f"{student_name} - {attendance_msg}",
            }
        )

    # Recognition error fallback
    return (
        jsonify(
            {
                "success": False,
                "recognized": False,
                "status": "error",
                "message": recognition_result.get("message", "Recognition failed"),
            }
        ),
        500,
    )


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Return JSON metrics for dashboard & portal."""
    metrics = get_dashboard_metrics()
    return jsonify(metrics)


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    """Administrator authentication."""
    if request.method == "POST":
        password = request.form.get("password", "")

        if password == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            flash("Logged in successfully.", "success")
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid administrator password. Please try again.", "danger")

    return render_template("admin_login.html")


@app.route("/admin")
@admin_required
def admin_dashboard():
    """Administrator dashboard."""
    metrics = get_dashboard_metrics()
    students = engine.student_names()
    recent_records = get_all_attendance_records()[:5]  # Latest 5 records
    return render_template(
        "admin.html",
        metrics=metrics,
        students=students,
        recent_records=recent_records,
    )


@app.route("/register", methods=["GET", "POST"])
@admin_required
def register_student():
    """
    Register a new student using browser camera capture:
    - Accepts student name and captured base64 image.
    - Validates that exactly ONE face is present.
    - Saves image to dataset/ as Firstname_Lastname.jpg.
    - Reloads FaceRecognitionEngine into memory.
    """
    if request.method == "POST":
        data = request.get_json(silent=True)
        if not data:
            return (
                jsonify({"success": False, "message": "No registration data received"}),
                400,
            )

        student_name = data.get("name", "").strip()
        base64_image = data.get("image", "")

        if not student_name:
            return (
                jsonify({"success": False, "message": "Student name cannot be empty"}),
                400,
            )

        sanitized_name = sanitize_student_name(student_name)
        if not sanitized_name:
            return (
                jsonify(
                    {"success": False, "message": "Invalid characters in student name"}
                ),
                400,
            )

        rgb_image = decode_base64_image(base64_image)
        if rgb_image is None:
            return (
                jsonify({"success": False, "message": "Failed to decode camera image"}),
                400,
            )

        # Validate that exactly ONE face exists in the captured frame
        is_valid, validation_msg, _ = engine.validate_registration_image(rgb_image)
        if not is_valid:
            return jsonify({"success": False, "message": validation_msg}), 400

        # Save image to dataset as Firstname_Lastname.jpg
        file_name = f"{sanitized_name.replace(' ', '_')}.jpg"
        save_path = os.path.join(DATASET_DIR, file_name)

        try:
            pil_image = Image.fromarray(rgb_image)
            pil_image.save(save_path, format="JPEG", quality=95)
            logger.info(f"Saved student image: {save_path}")

            # Reload engine so newly registered student is immediately recognized
            engine.reload()

            return jsonify(
                {
                    "success": True,
                    "message": "Student registered successfully.",
                    "name": sanitized_name,
                    "filename": file_name,
                }
            )
        except Exception as e:
            logger.error(f"Failed to save student image: {e}")
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"Error saving student image: {str(e)}",
                    }
                ),
                500,
            )

    return render_template("register.html")


@app.route("/attendance")
@admin_required
def view_attendance():
    """Display all attendance records in a clean table (newest first)."""
    records = get_all_attendance_records()
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_records = [r for r in records if r["date"] == today_str]

    return render_template(
        "attendance.html",
        records=records,
        total_count=len(records),
        today_count=len(today_records),
    )


@app.route("/reload_dataset")
@admin_required
def reload_dataset():
    """Explicitly trigger reload of face database into memory."""
    engine.reload()
    flash(
        f"Face database reloaded successfully ({engine.student_count()} students).",
        "info",
    )
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/logout")
def admin_logout():
    """Administrator logout."""
    session.pop("admin_logged_in", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("admin_login"))


# ==================================================
# APPLICATION ENTRYPOINT
# ==================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Run on all interfaces for local or container deployment
    app.run(host="0.0.0.0", port=port, debug=True)
