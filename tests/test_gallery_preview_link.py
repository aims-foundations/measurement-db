"""Preview reruns reuse existing share links without touching production aliases."""
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from scripts.ci import vercel_preview_link as links


class PreviewLinkTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, DEPLOY_ENV="preview", VERCEL_PROJECT_ID="project")
        environment.start()
        self.addCleanup(environment.stop)
        self.deployment = {"target": None, "readyState": "READY", "projectId": "project", "id": "new"}
        self.alias = {"uid": "alias", "projectId": "project", "deploymentId": "old"}
        self.calls = []

    def api(self, path, method="GET", payload=None):
        self.calls.append((path, method, payload))
        if path.startswith("/v13/deployments/"):
            return self.deployment
        if method == "PATCH":
            self.alias["protectionBypass"] = {"fresh": {"scope": "shareable-link", "expires": 9999999999}}
            if getattr(self, "conflict", False):
                raise HTTPError(path, 409, "concurrent creation", {}, None)
        return self.alias

    def test_expiry_uses_unix_seconds_from_the_vercel_api(self):
        # Real API shape: expires is seconds, while createdAt is milliseconds.
        alias = {"protectionBypass": {"existing": {"scope": "shareable-link",
                 "expires": 1792479239, "createdAt": 1791874439211}}}
        with patch.object(links.time, "time", return_value=1791874440):
            self.assertEqual(links.active_link(alias), "existing")
        with patch.object(links.time, "time", return_value=1792479239):
            self.assertIsNone(links.active_link(alias))

    def test_existing_share_link_is_reused_after_alias_update(self):
        self.alias["protectionBypass"] = {"existing": {"scope": "shareable-link"}}
        with patch.object(links, "api", side_effect=self.api):
            url = links.share("https://preview.vercel.app", 5)
        self.assertTrue(url.endswith("_vercel_share=existing"))
        self.assertEqual([method for _, method, _ in self.calls].count("POST"), 1)
        self.assertFalse(any(method == "PATCH" for _, method, _ in self.calls))

    def test_expired_link_is_rotated_and_concurrent_creation_is_safe(self):
        for conflict in (False, True):
            self.calls.clear()
            self.conflict = conflict
            self.alias["protectionBypass"] = {"expired": {"scope": "shareable-link", "expires": 1}}
            with patch.object(links, "api", side_effect=self.api):
                self.assertTrue(links.share("https://preview.vercel.app", 5).endswith("_vercel_share=fresh"))
            body = next(payload for _, method, payload in self.calls if method == "PATCH")
            self.assertEqual(body, {"ttl": 604800, "revoke": {"secret": "expired", "regenerate": True}})

    def test_production_deployment_or_unrelated_alias_cannot_be_changed(self):
        for change in ("production", "unrelated"):
            self.calls.clear()
            self.deployment["target"] = "production" if change == "production" else None
            self.alias["projectId"] = "different" if change == "unrelated" else "project"
            with patch.object(links, "api", side_effect=self.api), self.assertRaises(ValueError):
                links.share("https://preview.vercel.app", 5)
            self.assertTrue(all(method == "GET" for _, method, _ in self.calls))
