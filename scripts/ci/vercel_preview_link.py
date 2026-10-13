"""Create or reuse an expiring review link for this PR's preview alias only."""
import argparse
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


def api(path, method="GET", payload=None):
    url = "https://api.vercel.com" + path + "?" + urlencode({"teamId": os.environ["VERCEL_ORG_ID"]})
    with urlopen(Request(url, method=method, data=None if payload is None else json.dumps(payload).encode(),
                         headers={"Authorization": "Bearer " + os.environ["VERCEL_TOKEN"],
                                  "Content-Type": "application/json"}), timeout=60) as response:
        return json.load(response)


def active_link(alias):
    now = time.time() * 1000
    for token, entry in alias.get("protectionBypass", {}).items():
        if entry.get("scope") == "shareable-link" and entry.get("expires", float("inf")) > now:
            return token
    return None


def share(url: str, number: int) -> str:
    if number <= 0 or os.environ.get("DEPLOY_ENV") != "preview":
        raise ValueError("Review links are only created for PR previews")
    deployment = api("/v13/deployments/" + urlsplit(url).netloc)
    if (deployment.get("target") == "production" or deployment.get("readyState") != "READY"
            or deployment.get("projectId") != os.environ["VERCEL_PROJECT_ID"]):
        raise ValueError("Expected a ready preview in the configured project")
    hostname = f"measurement-db-pr-{number}.vercel.app"
    try:
        existing = api("/v4/aliases/" + hostname)
    except HTTPError as error:
        if error.code != 404:
            raise
    else:
        old = api("/v13/deployments/" + existing["deploymentId"])
        if existing.get("projectId") != os.environ["VERCEL_PROJECT_ID"] or old.get("target") == "production":
            raise ValueError("Refusing to replace an unrelated or production alias")
    api("/v2/deployments/" + deployment["id"] + "/aliases", "POST", {"alias": hostname})
    alias = api("/v4/aliases/" + hostname)
    token = active_link(alias)
    if not token:
        request = {"ttl": 604800}
        expired = next((key for key, entry in alias.get("protectionBypass", {}).items()
                        if entry.get("scope") == "shareable-link"), None)
        if expired:
            request["revoke"] = {"secret": expired, "regenerate": True}
        try:
            api("/aliases/" + alias["uid"] + "/protection-bypass", "PATCH", request)
        except HTTPError as error:
            if error.code != 409:
                raise
            # Another invocation may have created the link after our GET.
        token = active_link(api("/v4/aliases/" + hostname))
        if not token:
            raise ValueError("No active preview share link is available")
    return f"https://{hostname}/measurement-db?" + urlencode({"_vercel_share": token})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--pr", type=int, required=True)
    args = parser.parse_args()
    link = share(args.url, args.pr)
    # A review link is intentionally shareable; it contains no service token.
    print("Review preview: " + link)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(path).open("a") as stream:
            stream.write(f"\n[Review the prepared website]({link}) (link expires after seven days).\n")
