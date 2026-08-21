CATEGORY_KEYWORDS = {
    "accounts_access": [
        "password",
        "account",
        "login",
        "access request",
        "privileged access"
    ],
    "employee_software": [
        "software",
        "install",
        "workstation",
        "developer tools"
    ],
    "containers": [
        "docker",
        "container",
        "container platform"
    ],
    "infrastructure": [
        "gpu",
        "compute",
        "server"
    ],
    "monitoring_incidents": [
        "monitoring",
        "grafana",
        "prometheus",
        "alertmanager"
    ],
    "security": [
        "confidential",
        "restricted",
        "data handling",
        "personal cloud"
    ]
}

def route_query(question):
    question = question.lower()

    scores = {}

    # For every category:
    # count how many of its keywords appear in question

    # If exactly one category has the highest score:
    #     return that category
    #
    # If there's a tie or no match:
    #     return None

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in question:
                score += 1
        if score != 0:
            scores[category] = score

    if len(scores) == 0:
        return None
    
    highest_score = max(scores.values())
    highest_scorers = []

    for category, score in scores.items():
        if score == highest_score:
            highest_scorers.append(category)

    if len(highest_scorers) == 1:
        return highest_scorers[0]
    
    return None