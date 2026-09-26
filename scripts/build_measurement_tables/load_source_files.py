"""Load and validate raw benchmark source files."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatchcase
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import quote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen
from typing import Any


class SourceDataError(RuntimeError):
    """A downloaded source is absent, corrupt, or structurally unreadable."""


def github_tree_entries(repository: str, revision: str, paths: list[str] | None = None) -> list[dict]:
    """Read pinned GitHub files, optionally limiting traversal to named subtrees.

    Large result repositories can exceed GitHub's recursive-tree response limit.
    Path globs select complete subtrees without traversing unrelated screenshots;
    every selected file still carries the commit's native Git blob hash.
    """
    cache = {}
    headers = {"User-Agent": "measurement-db"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def read_tree(sha, recursive=False):
        key = sha, recursive
        if key not in cache:
            url = f"https://api.github.com/repos/{repository}/git/trees/{sha}"
            request = Request(url + ("?recursive=1" if recursive else ""), headers=headers)
            with urlopen(request, timeout=120) as response:
                tree = json.load(response)
            if tree.get("truncated"):
                raise SourceDataError("Upstream GitHub tree is truncated; select smaller tree_paths")
            cache[key] = tree["tree"]
        return cache[key]

    if paths is None:
        return read_tree(revision, recursive=True)
    if not paths or any(not isinstance(path, str) or "\\" in path or
                        any(part in ("", ".", "..", "**") for part in path.split("/")) for path in paths):
        raise SourceDataError("tree_paths must contain relative paths with component-wise globs")
    files = {}
    for pattern in paths:
        selected = [{"path": "", "sha": revision, "type": "tree"}]
        for component in pattern.split("/"):
            selected = [
                {**child, "path": f"{parent['path']}/{child['path']}".lstrip("/")}
                for parent in selected if parent["type"] == "tree"
                for child in read_tree(parent["sha"]) if fnmatchcase(child["path"], component)
            ]
        if not selected:
            raise SourceDataError(f"tree_paths matches no upstream path: {pattern}")
        for entry in selected:
            if entry["type"] == "blob":
                descendants = [entry]
            elif entry["type"] == "tree":
                descendants = [{**child, "path": entry["path"] + "/" + child["path"]}
                               for child in read_tree(entry["sha"], recursive=True) if child["type"] == "blob"]
            else:
                continue
            files.update({child["path"]: child for child in descendants})
    return sorted(files.values(), key=lambda entry: entry["path"])


def read_gpg_json(path: Path, *, password: str, scratch_dir: Path) -> Any:
    """Read a provider's password-encrypted JSON without extracting into raw/.

    GnuPG uses an isolated temporary home, never the user's keyring. Passwords
    for public benchmark releases belong in metadata alongside the source.
    """
    with tempfile.TemporaryDirectory(prefix=".gpg-", dir=scratch_dir) as home:
        try:
            result = subprocess.run(
                ["gpg", "--no-options", "--homedir", home, "--batch", "--no-tty",
                 "--pinentry-mode", "loopback", "--no-symkey-cache", "--passphrase-fd", "0",
                 "--decrypt", str(path)],
                input=(password + "\n").encode(), capture_output=True, check=True, timeout=120,
            )
        except FileNotFoundError as exc:
            raise SourceDataError("Install GnuPG (gpg) to read this encrypted upstream JSON") from exc
        except subprocess.CalledProcessError as exc:
            raise SourceDataError(f"Cannot decrypt {path}: {exc.stderr.decode(errors='replace')}") from exc
    return json.loads(result.stdout)


def html_index_entries(source: dict, named: dict, raw_dir: Path | None = None) -> list[dict]:
    """Resolve a static site's linked files and verify their complete content tree.

    The index and the selected page contents are both pinned. Existing raw files
    may supply the bytes, but are checked against the same tree fingerprint.
    This supports sites that publish transcripts without a repository archive.
    """
    name = source["name"]
    index = named.get(source["html_index"], {})
    if not {"url", "file", "size", "sha256"} <= index.keys():
        raise SourceDataError(f"{name}: HTML index must name a pinned HTTP source")

    def read(url, destination):
        if raw_dir is not None:
            path = raw_dir / destination
            if not path.resolve().is_relative_to(raw_dir.resolve()):
                raise SourceDataError(f"{name}: unsafe raw destination {destination}")
            if path.exists():
                return path.read_bytes()
        with urlopen(Request(url, headers={"User-Agent": "measurement-db", "Accept-Encoding": "identity"}), timeout=120) as response:
            return response.read()

    payload = read(index["url"], index["file"])
    if len(payload) != index["size"] or hashlib.sha256(payload).hexdigest() != index["sha256"]:
        raise SourceDataError(f"{name}: HTML index differs from its declared bytes")

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.hrefs = set()

        def handle_starttag(self, tag, attrs):
            if tag == "a" and (href := dict(attrs).get("href")):
                self.hrefs.add(href)

    parser = Links()
    parser.feed(payload.decode("utf-8"))
    base = urlparse(source["url"].rstrip("/") + "/")
    paths = {}
    for href in parser.hrefs:
        location = urlparse(urljoin(index["url"], href))
        if ((location.scheme, location.netloc) != (base.scheme, base.netloc)
                or not location.path.startswith(base.path) or location.query or location.fragment):
            continue
        relative = location.path.removeprefix(base.path)
        for rule in source["files"]:
            if match := re.fullmatch(rule["match"], relative):
                destination = rule["path"].format(path=relative, **match.groupdict())
                destination = re.sub(r"[^A-Za-z0-9._/-]", lambda m: f"_x{ord(m[0]):02x}_", destination)
                if Path(destination).is_absolute() or ".." in Path(destination).parts or destination in {"", "."}:
                    raise SourceDataError(f"{name}: unsafe raw destination {destination}")
                paths[relative] = (location.geturl(), destination)

    def inspect(relative):
        url, destination = paths[relative]
        content = read(url, destination)
        return dict(path=relative, size=len(content), digest=hashlib.sha256(content).hexdigest(),
                    hash_kind="sha256", url=url)

    with ThreadPoolExecutor(max_workers=4) as executor:
        entries = list(executor.map(inspect, sorted(paths)))
    identity = [{key: entry[key] for key in ("path", "size", "digest")} for entry in entries]
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if not entries or fingerprint != source.get("tree_sha256"):
        raise SourceDataError(f"{name}: linked page contents differ from the pinned tree ({fingerprint})")
    return entries


def json_index_entries(source: dict, named: dict, raw_dir: Path | None = None) -> list[dict]:
    """Resolve a JSON manifest to same-site files and pin the complete selection."""
    name, selector = source["name"], source["json_index"]
    index = named.get(selector["source"], {})
    if not {"url", "file", "size", "sha256"} <= index.keys():
        raise SourceDataError(f"{name}: JSON index must name a pinned HTTP source")

    def read(url, destination):
        if raw_dir is not None:
            path = raw_dir / destination
            if not path.resolve().is_relative_to(raw_dir.resolve()):
                raise SourceDataError(f"{name}: unsafe raw destination {destination}")
            if path.exists():
                return path.read_bytes()
        with urlopen(Request(url, headers={"User-Agent": "measurement-db", "Accept-Encoding": "identity"}), timeout=120) as response:
            return response.read()

    payload = read(index["url"], index["file"])
    if len(payload) != index["size"] or hashlib.sha256(payload).hexdigest() != index["sha256"]:
        raise SourceDataError(f"{name}: JSON index differs from its declared bytes")
    records = [json.loads(payload)]
    for field in selector["records"]:
        nested = []
        for record in records:
            if not isinstance(record, dict) or field not in record:
                raise SourceDataError(f"{name}: JSON index lacks records field {field}")
            value = record[field]
            nested.extend(value if isinstance(value, list) else [value])
        records = nested
    base = source["url"].rstrip("/") + "/"
    paths = {}
    for record in records:
        if not isinstance(record, dict):
            raise SourceDataError(f"{name}: JSON index records must be objects")
        try:
            relative = selector["path"].format_map(record)
        except (KeyError, ValueError, TypeError, AttributeError, IndexError) as exc:
            raise SourceDataError(f"{name}: invalid JSON index path template") from exc
        if not re.fullmatch(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*", relative) or any(
                part in {".", ".."} for part in relative.split("/")):
            raise SourceDataError(f"{name}: unsafe indexed source path {relative!r}")
        for rule in source["files"]:
            if match := re.fullmatch(rule["match"], relative):
                destination = rule["path"].format(path=relative, **match.groupdict())
                if Path(destination).is_absolute() or ".." in Path(destination).parts or destination in {"", "."}:
                    raise SourceDataError(f"{name}: unsafe raw destination {destination}")
                if relative in paths:
                    raise SourceDataError(f"{name}: duplicate indexed source path {relative}")
                paths[relative] = destination

    def inspect(relative):
        url = urljoin(base, relative)
        data = read(url, paths[relative])
        return dict(path=relative, size=len(data), digest=hashlib.sha256(data).hexdigest(),
                    hash_kind="sha256", url=url)

    with ThreadPoolExecutor(max_workers=4) as executor:
        entries = list(executor.map(inspect, sorted(paths)))
    identity = [{key: entry[key] for key in ("path", "size", "digest")} for entry in entries]
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if not entries or fingerprint != source.get("tree_sha256"):
        raise SourceDataError(f"{name}: JSON-indexed contents differ from the pinned tree ({fingerprint})")
    return entries


def google_drive_entries(source: dict, raw_dir: Path | None = None) -> list[dict]:
    """Pin public Drive folder membership and bytes without a per-file YAML inventory."""
    name = source['name']
    match = re.fullmatch(r'https://drive\.google\.com/drive/folders/([A-Za-z0-9_-]+)', source['url'])
    if match is None:
        raise SourceDataError(f'{name}: expected a public Google Drive folder URL')

    def read(url):
        with urlopen(Request(url, headers={'User-Agent': 'measurement-db', 'Accept-Encoding': 'identity'}), timeout=120) as response:
            return response.read()

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links, self.href, self.text = [], None, []

        def handle_starttag(self, tag, attrs):
            if tag == 'a':
                self.href, self.text = dict(attrs).get('href'), []

        def handle_data(self, data):
            if self.href is not None:
                self.text.append(data)

        def handle_endtag(self, tag):
            if tag == 'a' and self.href is not None:
                self.links.append((self.href, ''.join(self.text).strip()))
                self.href, self.text = None, []

    def children(entry):
        folder, prefix = entry
        parser = Links()
        parser.feed(read('https://drive.google.com/embeddedfolderview?id=' + folder).decode('utf-8'))
        rows = []
        for href, label in parser.links:
            folder_match = re.fullmatch(r'https://drive\.google\.com/drive/folders/([A-Za-z0-9_-]+)(?:\?.*)?', href)
            file_match = re.fullmatch(r'https://drive\.google\.com/file/d/([A-Za-z0-9_-]+)/view(?:\?.*)?', href)
            if folder_match is None and file_match is None:
                continue
            if not label or label in {'.', '..'} or '/' in label or '\\' in label:
                raise SourceDataError(f'{name}: unsafe Drive filename')
            rows.append(dict(id=(folder_match or file_match)[1], path=prefix + label,
                             kind='folder' if folder_match else 'file'))
        return rows

    pending, folders, files = [(match[1], '')], set(), {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        while pending:
            for folder, prefix in pending:
                if folder in folders:
                    raise SourceDataError(f'{name}: repeated folder or cycle in Drive tree')
                folders.add(folder)
            pages = list(executor.map(children, pending))
            pending = []
            for page in pages:
                for row in page:
                    if row['kind'] == 'folder':
                        pending.append((row['id'], row['path'] + '/'))
                    elif row['path'] in files:
                        raise SourceDataError(f'{name}: ambiguous duplicate Drive path')
                    else:
                        files[row['path']] = row['id']

    selected = {}
    for relative, file_id in files.items():
        for rule in source['files']:
            if match := re.fullmatch(rule['match'], relative):
                destination = rule['path'].format(path=relative, **match.groupdict())
                destination = re.sub(r'[^A-Za-z0-9._/-]', lambda m: f'_x{ord(m[0]):02x}_', destination)
                if Path(destination).is_absolute() or '..' in Path(destination).parts or destination in {'', '.'}:
                    raise SourceDataError(f'{name}: unsafe raw destination {destination}')
                selected[relative] = (file_id, destination)

    def inspect(relative):
        file_id, destination = selected[relative]
        path = raw_dir / destination if raw_dir is not None else None
        if path is not None and not path.resolve().is_relative_to(raw_dir.resolve()):
            raise SourceDataError(f'{name}: unsafe raw destination {destination}')
        url = 'https://drive.usercontent.google.com/download?' + urlencode(dict(id=file_id, export='download'))
        content = path.read_bytes() if path is not None and path.exists() else read(url)
        return dict(path=relative, drive_id=file_id, size=len(content), digest=hashlib.sha256(content).hexdigest(),
                    hash_kind='sha256', url=url)

    with ThreadPoolExecutor(max_workers=4) as executor:
        entries = list(executor.map(inspect, sorted(selected)))
    identity = [{key: entry[key] for key in ('path', 'drive_id', 'size', 'digest')} for entry in entries]
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if not entries or fingerprint != source.get('tree_sha256'):
        raise SourceDataError(f'{name}: Drive files differ from the pinned tree ({fingerprint})')
    return entries


def wandb_entries(source: dict) -> list[dict]:
    """Pin selected public W&B run files using stable URLs and provider checksums."""
    match = re.fullmatch(r"https://wandb\.ai/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)", source["url"])
    if match is None:
        raise SourceDataError("wandb_runs requires a public W&B project URL")
    entity, project = match.groups()
    selector = source["wandb_runs"]
    reference = re.compile(selector["latest_step"]) if "latest_step" in selector else None
    step_files = re.compile(selector["step_files"]) if reference else None
    if reference and ("step" not in reference.groupindex or "step" not in step_files.groupindex):
        raise SourceDataError("W&B checkpoint patterns must capture a named step group")

    def query(text, variables):
        request = Request("https://api.wandb.ai/graphql", data=json.dumps(dict(query=text,
            variables=dict(p=project, e=entity, **variables))).encode(),
            headers={"Content-Type":"application/json", "User-Agent":"measurement-db"})
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
        if result.get("errors") or not result.get("data", {}).get("project"):
            raise SourceDataError("W&B project query failed or is not publicly accessible")
        return result["data"]["project"]

    runs_query = """query R($p:String!,$e:String!,$c:String){project(name:$p,entityName:$e){
        runs(first:500,after:$c){pageInfo{hasNextPage endCursor}edges{node{name state}}}}}"""
    files_query = """query F($p:String!,$e:String!,$r:String!,$c:String){project(name:$p,entityName:$e){
        run(name:$r){files(first:500,after:$c){pageInfo{hasNextPage endCursor}edges{node{name sizeBytes md5}}}}}}"""
    runs, cursor, cursors = {}, None, set()
    while True:
        page = query(runs_query, dict(c=cursor))["runs"]
        for edge in page["edges"]:
            run = edge["node"]
            if run["name"] in runs:
                raise SourceDataError("W&B run inventory contains duplicate IDs")
            runs[run["name"]] = run
        if not page["pageInfo"]["hasNextPage"]: break
        cursor = page["pageInfo"]["endCursor"]
        if not cursor or cursor in cursors: raise SourceDataError("W&B run pagination did not advance")
        cursors.add(cursor)

    def files(run):
        nodes, cursor, cursors = {}, None, set()
        while True:
            page = query(files_query, dict(r=run["name"], c=cursor))["run"]["files"]
            for edge in page["edges"]:
                entry = edge["node"]
                if entry["name"] in nodes: raise SourceDataError("W&B file inventory contains duplicate paths")
                nodes[entry["name"]] = entry
            if not page["pageInfo"]["hasNextPage"]: break
            cursor = page["pageInfo"]["endCursor"]
            if not cursor or cursor in cursors: raise SourceDataError("W&B file pagination did not advance")
            cursors.add(cursor)
        steps = [int(match["step"]) for path in nodes if reference and (match := reference.fullmatch(path))]
        latest = max(steps) if steps else None
        selected = []
        for path, entry in nodes.items():
            if step_files and (step := step_files.fullmatch(path)) and int(step["step"]) != latest:
                continue
            relative = run["name"] + "/" + path
            if not any(re.fullmatch(rule["match"], relative) for rule in source["files"]): continue
            if any(part in ("", ".", "..") for part in relative.split("/")) or "\\" in relative:
                raise SourceDataError("W&B file path is not a safe relative path")
            try: digest = base64.b64decode(entry["md5"], validate=True)
            except (TypeError, ValueError) as exc: raise SourceDataError("W&B selected file lacks a valid checksum") from exc
            if len(digest) != 16 or not isinstance(entry["sizeBytes"], int) or entry["sizeBytes"] < 0:
                raise SourceDataError("W&B selected file has an invalid checksum or size")
            selected.append(dict(path=relative, size=entry["sizeBytes"], hash_kind="md5", digest=digest.hex(),
                url=f"https://api.wandb.ai/files/{entity}/{project}/" + quote(relative, safe="/")))
        return selected

    with ThreadPoolExecutor(max_workers=4) as executor:
        entries = [entry for group in executor.map(files, [run for run in runs.values()
            if run["state"] == selector["state"]]) for entry in group]
    entries.sort(key=lambda entry: entry["path"])
    identity = [{key:entry[key] for key in ("path", "size", "digest")} for entry in entries]
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if not entries or fingerprint != source.get("tree_sha256"):
        raise SourceDataError(f"W&B selected files differ from the pinned tree ({fingerprint})")
    return entries


def upstream_artifacts(sources: list[dict], names: tuple[str, ...], *, raw_dir: Path | None = None) -> list[dict]:
    """Resolve named upstream selections to pinned files and verify their inventory.

    Repository trees supply the file hashes. HTTP endpoints instead declare their
    expected bytes in metadata. Static HTML collections verify all selected page
    contents, reading existing raw files when available. No MeasurementDB archive
    is consulted.
    """
    named = {source["name"]: source for source in sources if "name" in source}
    if names == ("*",):
        names = tuple(named)
    if len(named) != sum("name" in source for source in sources):
        raise SourceDataError("Duplicate upstream source names")
    if not names or len(set(names)) != len(names) or set(names) - named.keys():
        raise SourceDataError(f"Select distinct declared upstream source names: {names}")
    artifacts, destinations = [], set()
    for name in names:
        source = named[name]
        url = source["url"]
        if source.get("git_lfs") and (urlparse(url).netloc != "github.com" or "files" not in source):
            raise SourceDataError(f"{name}: git_lfs requires a pinned GitHub file selection")
        if "tree_paths" in source and (urlparse(url).netloc != "github.com" or "files" not in source):
            raise SourceDataError(f"{name}: tree_paths requires a pinned GitHub file selection")
        if "file" in source:
            selected = [dict(file=source["file"], url=url, size=source["size"],
                             hash_kind="sha256", digest=source["sha256"])]
        else:
            location = urlparse(url)
            entries = []
            if "wandb_runs" in source:
                entries = wandb_entries(source)
            elif "json_index" in source:
                entries = json_index_entries(source, named, raw_dir)
            elif "html_index" in source:
                entries = html_index_entries(source, named, raw_dir)
            elif location.netloc == "drive.google.com":
                entries = google_drive_entries(source, raw_dir)
            elif location.netloc == "storage.googleapis.com" and "prefix" in source:
                bucket, prefix = location.path.strip("/"), source["prefix"]
                if not re.fullmatch(r"[a-z0-9._-]+", bucket):
                    raise SourceDataError(f"{name}: expected a public GCS bucket URL")
                prefixes = [prefix]
                if "helm_index" in source:
                    selector = source["helm_index"]
                    index = named.get(selector["source"], {})
                    if not {"url", "file", "size", "sha256"} <= index.keys():
                        raise SourceDataError(f"{name}: HELM index must name a pinned HTTP source")
                    request = Request(index["url"], headers={"User-Agent": "Mozilla/5.0"})
                    with urlopen(request, timeout=120) as response:
                        payload = response.read()
                    if len(payload) != index["size"] or hashlib.sha256(payload).hexdigest() != index["sha256"]:
                        raise SourceDataError(f"{name}: HELM release index differs from its declared bytes")
                    prefixes = []
                    for run in json.loads(payload):
                        if selector.get("group") and selector["group"] not in run.get("run_spec", {}).get("groups", []):
                            continue
                        path = run["run_path"]
                        marker = "benchmark_output/runs/"
                        if marker not in path:
                            raise SourceDataError(f"{name}: invalid HELM run path {path!r}")
                        relative = path[path.index(marker):].rstrip("/")
                        if ".." in Path(relative).parts:
                            raise SourceDataError(f"{name}: unsafe HELM run path {path!r}")
                        prefixes.append(prefix + relative + "/")
                    if not prefixes or len(set(prefixes)) != len(prefixes):
                        raise SourceDataError(f"{name}: HELM selection has no runs or duplicate run paths")
                for selected_prefix in prefixes:
                    parameters = {"prefix": selected_prefix, "maxResults": 1000,
                                  "fields": "items(name,size,generation,md5Hash,contentEncoding),nextPageToken"}
                    while True:
                        request = Request(f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?{urlencode(parameters)}",
                                          headers={"User-Agent": "measurement-db"})
                        with urlopen(request, timeout=120) as response:
                            page = json.load(response)
                        for entry in page.get("items", []):
                            relative = entry["name"].removeprefix(prefix)
                            if not any(re.fullmatch(rule["match"], relative) for rule in source["files"]):
                                continue
                            try:
                                checksum = base64.b64decode(entry["md5Hash"], validate=True)
                                if len(checksum) != 16 or not str(entry["generation"]).isdigit():
                                    raise ValueError("missing MD5 or generation")
                            except (KeyError, TypeError, ValueError) as exc:
                                raise SourceDataError(f"{name}: GCS object lacks a usable version/checksum: {entry['name']}") from exc
                            encoding = entry.get("contentEncoding", "")
                            if encoding not in ("", "gzip"):
                                raise SourceDataError(f"{name}: unsupported GCS content encoding {encoding!r}")
                            entries.append(dict(path=relative, size=int(entry["size"]), hash_kind="md5", digest=checksum.hex(),
                                generation=str(entry["generation"]), content_encoding=encoding,
                                url=f"https://storage.googleapis.com/{bucket}/{quote(entry['name'], safe='/')}?generation={entry['generation']}"))
                        if not page.get("nextPageToken"):
                            break
                        parameters["pageToken"] = page["nextPageToken"]
                identity = [{key: entry[key] for key in ("path", "generation", "size", "digest", "content_encoding")}
                            for entry in sorted(entries, key=lambda entry: entry["path"])]
                fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                if fingerprint != source.get("tree_sha256"):
                    raise SourceDataError(f"{name}: selected GCS objects differ from the pinned tree ({fingerprint})")
            else:
                revision = source["revision"]
                if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
                    raise SourceDataError(f"{name}: pin the upstream repository to a full commit SHA")
            if "html_index" in source or "json_index" in source or "wandb_runs" in source or location.netloc == "drive.google.com":
                pass
            elif location.netloc == "github.com":
                repository = location.path.strip("/")
                if len(repository.split("/")) != 2:
                    raise SourceDataError(f"{name}: expected a GitHub repository URL")
                for entry in github_tree_entries(repository, revision, source.get("tree_paths")):
                    if entry["type"] == "blob":
                        entries.append(dict(path=entry["path"], size=entry["size"],
                            hash_kind="git_sha1", digest=entry["sha"],
                            url=f"https://raw.githubusercontent.com/{repository}/{revision}/{quote(entry['path'], safe='/')}"))
            elif location.netloc == "huggingface.co" and location.path.startswith("/datasets/"):
                from huggingface_hub import HfApi, hf_hub_url
                repository = location.path.removeprefix("/datasets/").rstrip("/")
                if len(repository.split("/")) != 2:
                    raise SourceDataError(f"{name}: expected a Hugging Face dataset URL")
                for entry in HfApi().list_repo_tree(repository, repo_type="dataset", revision=revision, recursive=True):
                    if not hasattr(entry, "blob_id"):
                        continue
                    lfs = entry.lfs
                    digest = (lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256) if lfs else entry.blob_id
                    entries.append(dict(path=entry.path, size=entry.size,
                        hash_kind="sha256" if lfs else "git_sha1", digest=digest,
                        url=hf_hub_url(repository, entry.path, repo_type="dataset", revision=revision),
                        hf_repo=repository, hf_revision=revision, hf_path=entry.path))
            elif location.netloc != "storage.googleapis.com" or "prefix" not in source:
                raise SourceDataError(f"{name}: unsupported repository URL {url}")
            selected = []
            for rule in source["files"]:
                matches = 0
                for entry in sorted(entries, key=lambda row: row["path"]):
                    match = re.fullmatch(rule["match"], entry["path"])
                    if match is None:
                        continue
                    matches += 1
                    destination = rule["path"].format(path=entry["path"], **match.groupdict())
                    # Keep established cache filenames for punctuation in run names.
                    destination = re.sub(r"[^A-Za-z0-9._/-]", lambda m: f"_x{ord(m[0]):02x}_", destination)
                    if entry.get("content_encoding") == "gzip":
                        destination += ".gz"
                    if source.get("git_lfs"):
                        # The commit pins the pointer; its object ID pins the
                        # large file. Verify both, rather than saving the pointer.
                        request = Request(entry["url"], headers={"User-Agent": "measurement-db"})
                        with urlopen(request, timeout=120) as response:
                            pointer = response.read(1024)
                        digest = hashlib.sha1(f"blob {len(pointer)}\0".encode() + pointer).hexdigest()
                        if len(pointer) != entry["size"] or digest != entry["digest"]:
                            raise SourceDataError(f"{name}: Git LFS pointer differs from the pinned commit")
                        fields = re.fullmatch(rb"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n", pointer)
                        if fields is None:
                            raise SourceDataError(f"{name}: expected an unextended Git LFS v1 pointer: {entry['path']}")
                        entry = {**entry, "size": int(fields[2]), "hash_kind": "sha256", "digest": fields[1].decode(),
                                 "url": f"https://media.githubusercontent.com/media/{repository}/{revision}/{quote(entry['path'], safe='/')}"}
                    selected.append({**entry, "file": destination})
                if not matches:
                    raise SourceDataError(f"{name}: no upstream files match {rule['match']!r}")
        for artifact in selected:
            path = Path(artifact["file"])
            if path.is_absolute() or ".." in path.parts or str(path) in {".", ""}:
                raise SourceDataError(f"{name}: unsafe raw destination {path}")
            if path.as_posix() in destinations:
                raise SourceDataError(f"Duplicate upstream destination: {path}")
            destinations.add(path.as_posix())
            artifacts.append(artifact)
    return sorted(artifacts, key=lambda artifact: artifact["file"])


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SourceDataError(f"cannot read {path}: {exc}") from None
    return digest.hexdigest()


def verify_file(
    path: Path,
    *,
    minimum_size: int | None = None,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> Path:
    """Validate one source file and return its path.

    ``minimum_size`` is inclusive. Use ``expected_size`` when a provider's
    pinned byte count is known; supplying both is an authoring error.
    """
    if minimum_size is not None and expected_size is not None:
        raise ValueError("minimum_size and expected_size are mutually exclusive")
    for name, value in (
        ("minimum_size", minimum_size),
        ("expected_size", expected_size),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer or None")
    if expected_sha256 is not None and not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_sha256
    ):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal digits")

    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        raise SourceDataError(f"cannot stat {path}: {exc}") from None

    if minimum_size is not None and actual_size < minimum_size:
        raise SourceDataError(
            f"{path}: expected at least {minimum_size} bytes, found {actual_size}"
        )
    if expected_size is not None and actual_size != expected_size:
        raise SourceDataError(
            f"{path}: expected {expected_size} bytes, found {actual_size}"
        )
    if expected_sha256 is not None:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256.lower():
            raise SourceDataError(
                f"{path}: expected SHA-256 {expected_sha256.lower()}, "
                f"found {actual_sha256}"
            )
    return path


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    """Read nonblank JSONL lines, requiring one JSON object per physical line."""
    records: list[dict[str, Any]] = []
    line_number = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SourceDataError(
                        f"{path}:{line_number}:{exc.colno}: invalid JSON: {exc.msg}"
                    ) from None
                if not isinstance(value, dict):
                    raise SourceDataError(
                        f"{path}:{line_number}: expected a JSON object, "
                        f"found {type(value).__name__}"
                    )
                records.append(value)
    except SourceDataError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        location = f"{path}:{line_number}" if line_number else str(path)
        raise SourceDataError(f"cannot read {location}: {exc}") from None
    return records
