from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle, os
import numpy as np

# ---------- Paths (absolute) ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
TEMPLATES_DIR = os.path.join(BASE_DIR, "app", "templates")

app = FastAPI(title="Academic Advisor System")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ---------- Load model ----------
with open(os.path.join(MODEL_DIR, "academic_advisor_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

# ---------- Advising playbook ----------
ADVISING_PLAYBOOK = {
    "Intervention & Support": {
        "summary": "Your profile indicates you may be at academic risk. Immediate advisor contact is recommended.",
        "courses": [
            "Reduced-load schedule (12–14 credits)",
            "Repeat/grade-replacement for weak courses",
            "Study-skills workshop (1 credit)",
        ],
        "resources": [
            "Academic Success Center",
            "Tutoring Center",
            "Counseling Services",
            "Financial Aid Office",
        ],
        "next_steps": [
            "Book an emergency advising appointment this week",
            "Create a recovery GPA plan with your advisor",
            "Attend at least 2 tutoring sessions per week",
        ],
    },
    "Resource Referral": {
        "summary": "Your profile suggests non-academic barriers that may affect your studies.",
        "courses": [
            "Maintain current schedule",
            "Consider a light-load semester if stress is high",
        ],
        "resources": [
            "Counseling & Psychological Services",
            "Financial Aid / Emergency Grants",
            "Disability Accommodations Office",
            "Food & Housing Support",
        ],
        "next_steps": [
            "Contact the relevant office within 5 business days",
            "Inform your advisor of any accommodations you may need",
        ],
    },
    "Research & Grad School Track": {
        "summary": "You are well-positioned for research and postgraduate study.",
        "courses": [
            "Research Methods",
            "Advanced Statistics",
            "Undergraduate Thesis I & II",
            "Graduate-level elective",
        ],
        "resources": [
            "Office of Undergraduate Research",
            "Faculty Mentor Program",
            "Grad School Prep Workshops",
        ],
        "next_steps": [
            "Identify a faculty research mentor this semester",
            "Present at one undergraduate research symposium",
            "Prepare a CV and statement of purpose draft",
        ],
    },
    "AI/ML Career Track": {
        "summary": "Strong math + AI interest — you're set up for an AI/ML career.",
        "courses": [
            "Machine Learning",
            "Deep Learning",
            "Linear Algebra",
            "Probability & Statistics",
            "Data Structures & Algorithms",
        ],
        "resources": [
            "AI/ML Student Club",
            "Kaggle / Open Source Projects",
            "Career Services — Tech Track",
        ],
        "next_steps": [
            "Complete 2 end-to-end ML projects on GitHub",
            "Apply for a summer ML internship",
            "Attend one AI conference or hackathon",
        ],
    },
    "Data Science Career Track": {
        "summary": "Your profile fits data science and analytics roles.",
        "courses": [
            "Data Mining",
            "Statistical Inference",
            "Data Visualization",
            "SQL & Databases",
            "Python for Data Science",
        ],
        "resources": [
            "Data Science Club",
            "Analytics Internships",
            "Career Services — Analytics Track",
        ],
        "next_steps": [
            "Build a portfolio of 3 data projects",
            "Earn one cloud/analytics certification",
            "Network at one analytics meetup per month",
        ],
    },
    "Web Development Career Track": {
        "summary": "Programming strength + web interest — a strong fit for web development.",
        "courses": [
            "Web Development",
            "Database Systems",
            "Software Engineering",
            "Human-Computer Interaction",
            "Cloud Computing",
        ],
        "resources": [
            "Web Dev Club",
            "Freelance / Open Source Opportunities",
            "Career Services — Software Track",
        ],
        "next_steps": [
            "Build and deploy 2 full-stack apps",
            "Contribute to one open-source repo",
            "Apply for a frontend/backend internship",
        ],
    },
    "Foundational Course Planning": {
        "summary": "You're early in your degree — build a solid foundation now.",
        "courses": [
            "Intro to Programming",
            "College Writing",
            "Calculus I",
            "General Education electives",
            "Study Skills (1 credit)",
        ],
        "resources": [
            "First-Year Experience Office",
            "Peer Advising Program",
            "Learning Center",
        ],
        "next_steps": [
            "Meet your advisor before registration opens",
            "Map out a 4-year plan (even a rough one)",
            "Explore 2–3 possible majors this year",
        ],
    },
    "Major Declaration Guidance": {
        "summary": "Time to commit to a major — explore, then declare.",
        "courses": [
            "Intro courses for 2 prospective majors",
            "Career Exploration seminar",
            "General Education electives",
        ],
        "resources": [
            "Career Services — Major Exploration",
            "Faculty Mentors in each department",
            "Peer Advising Program",
        ],
        "next_steps": [
            "Attend 2 department info sessions",
            "Shadow an upperclassman in your top-choice major",
            "Declare by the end of this semester",
        ],
    },
    "Academic Planning": {
        "summary": "You're on track — keep building your semester-by-semester plan.",
        "courses": [
            "Major core requirement",
            "Minor / elective",
            "General Education",
            "1 skills course",
        ],
        "resources": [
            "Degree Audit Tool",
            "Advisor Office Hours",
            "Career Services",
        ],
        "next_steps": [
            "Review your degree audit before registration",
            "Confirm your minor / concentration choice",
            "Set one academic goal for this semester",
        ],
    },
}

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": None},
    )


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    gpa: float = Form(...),
    year: int = Form(...),
    credits_completed: int = Form(...),
    math_score: int = Form(...),
    programming_score: int = Form(...),
    communication_score: int = Form(...),
    interest_ai: int = Form(0),
    interest_web: int = Form(0),
    interest_data: int = Form(0),
    interest_research: int = Form(0),
    on_probation: int = Form(0),
    financial_need: int = Form(0),
    mental_health_flag: int = Form(0),
    at_risk: int = Form(0),
):
    features = np.array([[
        gpa, year, credits_completed,
        math_score, programming_score, communication_score,
        interest_ai, interest_web, interest_data, interest_research,
        on_probation, financial_need, mental_health_flag, at_risk,
    ]])

    pred = model.predict(features)[0]
    action = le.inverse_transform([pred])[0]
    playbook = ADVISING_PLAYBOOK.get(action, {
        "summary": "General advising guidance.",
        "courses": ["Meet with your advisor to plan courses"],
        "resources": ["Academic Advising Office"],
        "next_steps": ["Book an appointment with your advisor"],
    })

    proba = model.predict_proba(features)[0]
    confidence = round(float(np.max(proba)) * 100, 1)

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "action": action,
            "confidence": confidence,
            "playbook": playbook,
        },
    )


# ---------- Auto-open browser when run directly ----------
if __name__ == "__main__":
    import threading, webbrowser
    import uvicorn

    def _open():
        webbrowser.open("http://127.0.0.1:8000")

    threading.Timer(1.5, _open).start()
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)