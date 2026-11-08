"""Validate one upstream merge before updating the fork through a pull request."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

REPOSITORY = "kevinbaijnath/localthings"
API = f"repos/{REPOSITORY}"
UPSTREAM = "https://github.com/mbillow/localthings.git"
AUTHOR_NAME = "Kevin Baijnath"
AUTHOR_EMAIL = "12387485+kevinbaijnath@users.noreply.github.com"
ACTIONS_APP_ID = 15368
REQUIRED_CHECKS = frozenset(
    {"Hassfest validation", "HACS validation", "Lint, format, and type check", "Pytest"}
)


def sanitize(text: str) -> str:
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        if token := os.environ.get(name):
            text = text.replace(token, "***")
    return text


def git(root: Path, *args: str, env: dict[str, str] | None = None) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        print(sanitize(f"Git {args[0]} failed (exit {error.returncode})"), file=sys.stderr)
        print(sanitize(error.stdout + error.stderr), file=sys.stderr)
        raise


def prepare(root: Path, upstream: str, branch: str, report: dict) -> dict[str, str] | None:
    base = git(root, "rev-parse", "HEAD")
    report.update(base=base, head=base, stage="fetch")
    git(root, "fetch", "--no-tags", upstream, "main")
    upstream_sha = git(root, "rev-parse", "FETCH_HEAD")
    report.update(upstream=upstream_sha, stage="merge")
    ancestor = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", upstream_sha, base],
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode == 0:
        return None
    if ancestor.returncode != 1:
        ancestor.check_returncode()
    git(root, "switch", "--create", branch)
    identity = {
        **os.environ,
        "GIT_AUTHOR_NAME": AUTHOR_NAME,
        "GIT_AUTHOR_EMAIL": AUTHOR_EMAIL,
        "GIT_COMMITTER_NAME": AUTHOR_NAME,
        "GIT_COMMITTER_EMAIL": AUTHOR_EMAIL,
    }
    try:
        git(root, "merge", "--no-ff", "--no-edit", upstream_sha, env=identity)
    except subprocess.CalledProcessError:
        conflicts = git(root, "diff", "--name-only", "--diff-filter=U").splitlines()
        if conflicts:
            report.update(error_kind="merge_conflict", conflicts=conflicts)
        raise
    candidate = {
        "base": base,
        "upstream": upstream_sha,
        "head": git(root, "rev-parse", "HEAD"),
        "branch": branch,
    }
    report.update(candidate)
    return candidate


class GitHub:
    """Keep credentials in gh's environment instead of command arguments."""

    @staticmethod
    def api(path: str, method: str = "GET", data: dict | None = None):
        args = ["gh", "api", path, "--method", method]
        if data is not None:
            args.extend(["--input", "-"])
        result = subprocess.run(
            args,
            input=json.dumps(data) if data is not None else None,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout) if result.stdout.strip() else None


def validate(github, candidate: dict[str, str], *, timeout: int = 1800) -> dict:
    workflow = github.api(f"{API}/actions/workflows/validate.yml")
    if workflow["state"] != "active" or workflow["path"] != ".github/workflows/validate.yml":
        raise RuntimeError("The validation workflow is not active")
    workflow_id = workflow["id"]
    github.api(
        f"{API}/actions/workflows/{workflow_id}/dispatches",
        "POST",
        {"ref": candidate["branch"]},
    )
    query = urlencode(
        {"event": "workflow_dispatch", "branch": candidate["branch"], "per_page": 100}
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        runs = github.api(f"{API}/actions/workflows/{workflow_id}/runs?{query}")["workflow_runs"]
        matches = [
            run
            for run in runs
            if run["workflow_id"] == workflow_id
            and run["head_sha"] == candidate["head"]
            and run["head_branch"] == candidate["branch"]
            and run["event"] == "workflow_dispatch"
        ]
        if len(matches) > 1:
            raise RuntimeError("More than one validation run matches the candidate")
        if matches and matches[0]["status"] == "completed":
            run = matches[0]
            if run["conclusion"] != "success":
                raise RuntimeError("Candidate validation failed")
            jobs = github.api(f"{API}/actions/runs/{run['id']}/jobs?per_page=100")
            passed = {job["name"] for job in jobs["jobs"] if job["conclusion"] == "success"}
            if (
                not REQUIRED_CHECKS.issubset(passed)
                or len(jobs["jobs"]) != jobs["total_count"]
                or any(job["conclusion"] != "success" for job in jobs["jobs"])
            ):
                raise RuntimeError("Candidate mandatory checks did not all pass")
            return run
        time.sleep(10)
    raise RuntimeError("Candidate validation did not finish before the deadline")


def verify_protection(github) -> None:
    rules = github.api(f"{API}/rules/branches/main?per_page=100")
    strict_checks = [
        rule["parameters"]
        for rule in rules
        if rule["type"] == "required_status_checks"
        and rule["parameters"].get("strict_required_status_checks_policy") is True
    ]
    bound = {
        check["context"]
        for policy in strict_checks
        for check in policy["required_status_checks"]
        if check.get("integration_id") == ACTIONS_APP_ID
    }
    types = {rule["type"] for rule in rules}
    if (
        len(rules) == 100
        or not REQUIRED_CHECKS.issubset(bound)
        or not {"non_fast_forward", "deletion"}.issubset(types)
    ):
        raise RuntimeError("Required active main branch rules are absent or incomplete")


def verify_revisions(github, candidate: dict[str, str]) -> None:
    base = github.api(f"{API}/git/ref/heads/main")["object"]["sha"]
    head = github.api(f"{API}/git/ref/heads/{candidate['branch']}")["object"]["sha"]
    if base != candidate["base"] or head != candidate["head"]:
        raise RuntimeError("The candidate head or main base changed; promotion stopped")


def promote(github, candidate: dict[str, str]) -> str:
    verify_protection(github)
    verify_revisions(github, candidate)
    pull = github.api(
        f"{API}/pulls",
        "POST",
        {
            "base": "main",
            "head": candidate["branch"],
            "title": f"Merge upstream LocalThings through {candidate['upstream'][:12]}",
            "body": (
                "Merge upstream LocalThings changes while preserving the fork's Frame TV support. "
                "The existing validation workflow passed all mandatory checks on this exact "
                f"candidate commit: {candidate['head']}. "
                "This updates source code only; it does not install or restart Home Assistant."
            ),
        },
    )
    verify_revisions(github, candidate)
    result = github.api(
        f"{API}/pulls/{pull['number']}/merge",
        "PUT",
        {"sha": candidate["head"], "merge_method": "merge"},
    )
    if result["merged"] is not True:
        raise RuntimeError("GitHub refused the protected pull request merge")
    return pull["html_url"]


def main() -> None:
    report = {"result": "failed", "stage": "context", "conflicts": []}
    try:
        if (
            os.environ["GITHUB_REPOSITORY"] != REPOSITORY
            or os.environ["GITHUB_REF"] != "refs/heads/main"
        ):
            raise RuntimeError("This updater runs only from the maintained fork's main branch")
        branch = (
            f"codex/upstream-sync-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
        )
        report.update(branch=branch, stage="prepare")
        root = Path.cwd()
        candidate = prepare(root, UPSTREAM, branch, report)
        if candidate is None:
            report.update(result="up_to_date", next_action="No update is needed.")
        else:
            report["stage"] = "push"
            git(root, "push", "origin", f"{candidate['head']}:refs/heads/{branch}")
            github = GitHub()
            report["stage"] = "validation"
            run = validate(github, candidate)
            report.update(validation_run=run["html_url"], stage="promotion")
            pull_url = promote(github, candidate)
            report.update(
                result="merged",
                pr=pull_url,
                next_action="Source updated. Home Assistant was not installed or restarted.",
            )
        report["stage"] = "complete"
    except Exception as error:
        if "error_kind" not in report:
            report["error_kind"] = (
                "command_failed"
                if isinstance(error, subprocess.CalledProcessError)
                else "sync_failed"
            )
        report["error"] = sanitize(str(error))
        if report["error_kind"] == "merge_conflict":
            report["next_action"] = (
                "Resolve the listed files against these revisions, then run all required checks. "
                "The candidate was not pushed."
            )
        elif report["stage"] in {"context", "prepare", "fetch", "merge"}:
            report["next_action"] = "Fix the reported command or configuration error, then retry."
        elif report["stage"] == "push":
            report["next_action"] = "Check repository access and the push error, then retry."
        elif report["stage"] == "validation":
            report["next_action"] = "Inspect candidate validation and fix it before promotion."
        else:
            report["next_action"] = (
                "Inspect the pull request, branch rules and revisions before retrying promotion."
            )
        if isinstance(error, subprocess.CalledProcessError):
            report["diagnostics"] = sanitize((error.stdout or "") + (error.stderr or ""))
            if error.cmd[0] == "gh":
                print(report["diagnostics"], file=sys.stderr)
        raise
    finally:
        rendered = sanitize(json.dumps(report, indent=2))
        print(rendered)
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as summary:
            summary.write("## LocalThings upstream sync\n\n```json\n")
            summary.write(rendered + "\n```\n")


if __name__ == "__main__":
    main()
