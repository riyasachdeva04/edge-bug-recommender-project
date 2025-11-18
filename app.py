from flask import Flask, render_template, request, redirect, url_for
from recommender import BugRecommender

app = Flask(__name__)
recommender = BugRecommender(data_path="data/bugs.csv")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    try:
        # When user enters bug ID (POST)
        if request.method == "POST":
            bug_id = int(request.form["bug_id"])
        # When returning from feedback (GET)
        else:
            bug_id = int(request.args.get("bug_id"))

        bug = recommender.get_bug(bug_id)
        recs = recommender.recommend_by_id(bug_id)

        return render_template("index.html", recs=recs, bug=bug)

    except (KeyError, ValueError, IndexError, TypeError):
        return redirect(url_for("home"))

@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        original_bug_id = int(request.form["bug_id"])
        rec_id = int(request.form["rec_id"])
        reward = int(request.form["reward"])

        recommender.update_feedback(original_bug_id, rec_id, reward)

        # REDIRECT BACK TO SAME BUG ID
        return redirect(url_for("recommend") + f"?bug_id={original_bug_id}")

    except (KeyError, ValueError, IndexError):
        return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
