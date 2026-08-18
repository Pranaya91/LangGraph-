import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional

from flask import Flask, request, render_template_string

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================================================
# 1. GEMINI API KEY
# =========================================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set.")


# =========================================================
# 2. LLM INITIALIZATION
# =========================================================

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key
)

llm = llm_flash


# =========================================================
# 3. STATE DEFINITION
# =========================================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# =========================================================
# 4. TOOLS
# =========================================================

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return output or error."""

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()

    sys.stdout = new_stdout

    try:
        local_scope = {}

        exec(clean_code, {}, local_scope)

        result = new_stdout.getvalue()

    except Exception:
        result = f"Execution Error:\n{traceback.format_exc()}"

    finally:
        sys.stdout = old_stdout

    return (
        result.strip()
        if result.strip()
        else "Success (no terminal output)"
    )


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate test scenarios for a coding task."""

    prompt = (
        "You are a Senior QA Engineer. "
        "Generate 3 to 5 highly specific test scenarios "
        f"for the following coding task: '{task_description}'. "
        "Include standard cases and edge cases. "
        "Return them as a numbered list."
    )

    response = llm.invoke(prompt)

    return (
        response.content
        if hasattr(response, "content")
        else str(response)
    )


# =========================================================
# 5. GRAPH NODES
# =========================================================

# ---------------- INPUT NODE ----------------

def task_input_node(state: CrewState):

    task = state["messages"][-1].content

    print("\n[INPUT]")
    print("Task:", task)

    return {
        "messages": [
            HumanMessage(content=task)
        ],
        "next_step": "developer"
    }


# ---------------- DEVELOPER NODE ----------------

def real_time_developer(state: CrewState):

    print("\n[DEVELOPER] Generating code...")

    task = state["messages"][-1].content

    dev_prompt = (
        f"Write a clean Python script to solve this: {task}. "
        "The code will be automatically executed by a tester. "
        "Do NOT use input() or interactive user input. "
        "Use sample values directly in the code so it can run automatically. "
        "Include print statements to show the results. "
        "Only return the code, no explanation or markdown formatting."
    )

    response = llm_flash.invoke(dev_prompt)

    content = response.content

    if isinstance(content, list):

        code_parts = []

        for item in content:

            if isinstance(item, dict) and "text" in item:
                code_parts.append(item["text"])

            elif isinstance(item, str):
                code_parts.append(item)

        code_str = "\n".join(code_parts)

    else:

        code_str = str(content)

    print("\n[DEVELOPER] Generated code:")
    print(code_str)

    return {
        "code": code_str,
        "next_step": "tester"
    }


# ---------------- TESTER NODE ----------------

def real_time_tester(state: CrewState):

    print("\n[TESTER] Generating test cases...")

    task = state["messages"][-1].content

    # Generate test scenarios
    test_cases = generate_test_cases.invoke(task)

    if isinstance(test_cases, list):

        cases_parts = []

        for item in test_cases:

            if isinstance(item, dict) and "text" in item:
                cases_parts.append(item["text"])

            elif isinstance(item, str):
                cases_parts.append(item)

        cases_str = "\n".join(cases_parts)

    else:

        cases_str = str(test_cases)

    # Execute generated code
    print("\n[TESTER] Executing generated code...")

    execution_result = run_python_code.invoke(
        {
            "code": state["code"]
        }
    )

    report = (
        "### EXECUTION OUTPUT:\n"
        f"{execution_result}\n\n"
        "### TEST SCENARIOS:\n"
        f"{cases_str}"
    )

    print("\n[TESTER] Testing completed.")

    return {
        "report": report,
        "next_step": "manager"
    }


# ---------------- MANAGER NODE ----------------

def manager_decision_node(state: CrewState):

    print("\n[MANAGER] Reviewing test report...")

    print(state.get(
        "report",
        "No report available."
    ))

    print("\n[MANAGER] Report reviewed successfully.")
    print("[MANAGER] Sending task to Archiver...")

    return {
        "next_step": "archiver"
    }


# ---------------- ARCHIVER NODE ----------------

def archiver_node(state: CrewState):

    print("\n[ARCHIVER] Task stored successfully.")
    print("[ARCHIVER] Closing workflow.")

    return {
        "next_step": "exit"
    }


# =========================================================
# 6. LANGGRAPH CONSTRUCTION
# =========================================================

rt_workflow = StateGraph(CrewState)


# Add nodes

rt_workflow.add_node(
    "task_input",
    task_input_node
)

rt_workflow.add_node(
    "developer",
    real_time_developer
)

rt_workflow.add_node(
    "tester",
    real_time_tester
)

rt_workflow.add_node(
    "manager_decision",
    manager_decision_node
)

rt_workflow.add_node(
    "archiver",
    archiver_node
)


# START → INPUT

rt_workflow.add_edge(
    START,
    "task_input"
)


# INPUT → DEVELOPER

def route_from_input(state: CrewState):

    if state.get("next_step") == "exit":
        return END

    return "developer"


rt_workflow.add_conditional_edges(
    "task_input",
    route_from_input
)


# DEVELOPER → TESTER

rt_workflow.add_edge(
    "developer",
    "tester"
)


# TESTER → MANAGER

rt_workflow.add_edge(
    "tester",
    "manager_decision"
)


# MANAGER → ARCHIVER

def route_from_decision(state: CrewState):

    if state.get("next_step") == "archiver":
        return "archiver"

    return "task_input"


rt_workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision
)


# ARCHIVER → END

rt_workflow.add_edge(
    "archiver",
    END
)


# Compile graph

rt_app = rt_workflow.compile()

print(
    "LangGraph pipeline compiled and ready."
)


# =========================================================
# 7. FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# HTML PAGE
# =========================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

    <title>LangGraph AI Developer</title>

    <style>

        body {
            font-family: Arial, sans-serif;
            background: #f4f4f4;
            margin: 0;
            padding: 40px;
        }

        .container {
            max-width: 900px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        h1 {
            text-align: center;
            color: #333;
        }

        textarea {
            width: 100%;
            height: 120px;
            padding: 12px;
            font-size: 16px;
            box-sizing: border-box;
            border-radius: 8px;
            border: 1px solid #ccc;
        }

        button {
            margin-top: 15px;
            padding: 12px 25px;
            background: #6c5ce7;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
        }

        button:hover {
            background: #5848c2;
        }

        .flow {
            margin: 20px 0;
            padding: 15px;
            background: #eee;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
        }

        .result {
            margin-top: 30px;
            background: #111;
            color: #eee;
            padding: 20px;
            border-radius: 10px;
            white-space: pre-wrap;
            overflow-x: auto;
            line-height: 1.5;
        }

    </style>

</head>


<body>

<div class="container">

    <h1>
        🤖 LangGraph AI Developer & Tester
    </h1>


    <div class="flow">

        📥 INPUT
        →
        👨‍💻 DEVELOPER
        →
        🧪 TESTER
        →
        👨‍💼 MANAGER
        →
        🗄️ ARCHIVER
        →
        🏁 END

    </div>


    <form method="POST">

        <label>
            <b>Enter your coding task:</b>
        </label>

        <br><br>

        <textarea
            name="task"
            placeholder="Example: Write a Python program to check whether a number is prime."
            required
        ></textarea>

        <br>

        <button type="submit">
            Generate
        </button>

    </form>


    {% if report %}

        <hr>

        <div class="result">

            {{ report }}

        </div>

    {% endif %}

</div>

</body>

</html>
"""


# =========================================================
# 8. WEB ROUTE
# =========================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)

def home():

    report = None

    if request.method == "POST":

        task = request.form.get(
            "task",
            ""
        ).strip()

        if task:

            try:

                initial_state = {

                    "messages": [
                        HumanMessage(
                            content=task
                        )
                    ],

                    "next_step": None,

                    "code": None,

                    "report": None
                }


                # Run LangGraph

                result = rt_app.invoke(
                    initial_state,
                    config={
                        "recursion_limit": 50
                    }
                )


                generated_code = result.get(
                    "code",
                    "No code generated."
                )


                tester_report = result.get(
                    "report",
                    "No report generated."
                )


                # =================================================
                # FINAL WEBSITE OUTPUT
                # =================================================

                report = (

                    "📥 INPUT\n\n"

                    f"{task}\n\n"


                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"


                    "👨‍💻 DEVELOPER\n\n"

                    "The Developer node generated the following code:\n\n"

                    f"{generated_code}\n\n"


                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"


                    "🧪 TESTER\n\n"

                    "The Tester node generated test scenarios "
                    "and executed the generated code.\n\n"

                    f"{tester_report}\n\n"


                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"


                    "👨‍💼 MANAGER\n\n"

                    "Report reviewed successfully.\n"

                    "Decision: Send task to Archiver.\n\n"


                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"


                    "🗄️ ARCHIVER\n\n"

                    "Task stored successfully.\n"

                    "Workflow is closing.\n\n"


                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"


                    "🏁 END\n\n"

                    "Workflow completed successfully.\n\n"


                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"


                    "🔄 LANGGRAPH WORKFLOW\n\n"

                    "INPUT → DEVELOPER → TESTER → "
                    "MANAGER → ARCHIVER → END"
                )


            except Exception:

                report = (
                    "❌ WORKFLOW ERROR\n\n"
                    f"{traceback.format_exc()}"
                )


    return render_template_string(
        HTML,
        report=report
    )


# =========================================================
# 9. START FLASK SERVER
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
