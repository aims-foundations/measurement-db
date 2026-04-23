"""Build AndroidWorld long-form responses from publicly available per-task data.

AndroidWorld (Google Research, ICLR 2025) evaluates AI agents on 116
programmatic tasks across 20 Android apps.

Data sources:
  - Per-task pass/fail scraped from 3 public agent benchmark pages
    (DroidRun, FinalRun, AutoDevice). Each reports a list of failed tasks.
  - Canonical task list derived from paper Appendix F and gbox.ai trajectory
    filenames (116 tasks).

Encoding:
    Subject:  agent label (e.g. "DroidRun", "FinalRun", "AutoDevice").
    Item:     one item per canonical task name (116 tasks).
    Response: 1 = pass, 0 = fail.
    trial=1 (single reported run per agent per task on these pages).

Outputs:
  - responses.parquet
  - _contrib/{subjects,items,benchmarks}.parquet
"""

INFO = {
    'description': "AndroidWorld: 116-task Android agent benchmark; per-task data from 3 public agent pages",
    'testing_condition': 'per-task binary pass/fail; aggregate-only leaderboard entries are not emitted here',
    'paper_url': 'https://arxiv.org/abs/2405.14573',
    'data_source_url': 'https://docs.google.com/spreadsheets/d/1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo',
    'subject_type': 'agent',
    'item_type': 'task',
    'license': 'Apache-2.0',
    'citation': """@misc{rawles2025androidworlddynamicbenchmarkingenvironment,
      title={AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents},
      author={Christopher Rawles and Sarah Clinckemaillie and Yifan Chang and Jonathan Waltz and Gabrielle Lau and Marybeth Fair and Alice Li and William Bishop and Wei Li and Folawiyo Campbell-Ajala and Daniel Toyama and Robert Berry and Divya Tyamagundlu and Timothy Lillicrap and Oriana Riva},
      year={2025},
      eprint={2405.14573},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2405.14573},
}""",
    'tags': ['agent'],
}


import sys
from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
CONTRIB_DIR = _BENCHMARK_DIR / "_contrib"
RESPONSES_PATH = _BENCHMARK_DIR / "responses.parquet"
RAW_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_BENCHMARK_DIR.parent))
from _registry import (  # noqa: E402
    get_benchmark_id, register_item, resolve_subject, save as registry_save,
    ensure_unique_trials,
)


CANONICAL_TASKS = sorted([
    "AudioRecorderRecordAudio", "AudioRecorderRecordAudioWithFileName",
    "BrowserDraw", "BrowserMaze", "BrowserMultiply",
    "CameraTakePhoto", "CameraTakeVideo",
    "ClockStopWatchPausedVerify", "ClockStopWatchRunning", "ClockTimerEntry",
    "ContactsAddContact", "ContactsNewContactDraft",
    "ExpenseAddMultiple", "ExpenseAddMultipleFromGallery",
    "ExpenseAddMultipleFromMarkor", "ExpenseAddSingle",
    "ExpenseDeleteDuplicates", "ExpenseDeleteDuplicates2",
    "ExpenseDeleteMultiple", "ExpenseDeleteMultiple2", "ExpenseDeleteSingle",
    "FilesDeleteFile", "FilesMoveFile",
    "MarkorAddNoteHeader", "MarkorChangeNoteContent", "MarkorCreateFolder",
    "MarkorCreateNote", "MarkorCreateNoteAndSms",
    "MarkorCreateNoteFromClipboard", "MarkorDeleteAllNotes",
    "MarkorDeleteNewestNote", "MarkorDeleteNote", "MarkorEditNote",
    "MarkorMergeNotes", "MarkorMoveNote", "MarkorTranscribeReceipt",
    "MarkorTranscribeVideo",
    "NotesIsTodo", "NotesMeetingAttendeeCount", "NotesRecipeIngredientCount",
    "NotesTodoItemCount",
    "OpenAppTaskEval", "OsmAndFavorite", "OsmAndMarker", "OsmAndTrack",
    "RecipeAddMultipleRecipes", "RecipeAddMultipleRecipesFromImage",
    "RecipeAddMultipleRecipesFromMarkor", "RecipeAddMultipleRecipesFromMarkor2",
    "RecipeAddSingleRecipe", "RecipeDeleteDuplicateRecipes",
    "RecipeDeleteDuplicateRecipes2", "RecipeDeleteDuplicateRecipes3",
    "RecipeDeleteMultipleRecipes",
    "RecipeDeleteMultipleRecipesWithConstraint",
    "RecipeDeleteMultipleRecipesWithNoise", "RecipeDeleteSingleRecipe",
    "RecipeDeleteSingleWithRecipeWithNoise",
    "RetroCreatePlaylist", "RetroPlayingQueue", "RetroPlaylistDuration",
    "RetroSavePlaylist",
    "SaveCopyOfReceiptTaskEval",
    "SimpleCalendarAddOneEvent", "SimpleCalendarAddOneEventInTwoWeeks",
    "SimpleCalendarAddOneEventRelativeDay", "SimpleCalendarAddOneEventTomorrow",
    "SimpleCalendarAddRepeatingEvent", "SimpleCalendarAnyEventsOnDate",
    "SimpleCalendarDeleteEvents", "SimpleCalendarDeleteEventsOnRelativeDay",
    "SimpleCalendarDeleteOneEvent", "SimpleCalendarEventOnDateAtTime",
    "SimpleCalendarEventsInNextWeek", "SimpleCalendarEventsInTimeRange",
    "SimpleCalendarEventsOnDate", "SimpleCalendarFirstEventAfterStartTime",
    "SimpleCalendarLocationOfEvent", "SimpleCalendarNextEvent",
    "SimpleCalendarNextMeetingWithPerson",
    "SimpleDrawProCreateDrawing",
    "SimpleSmsReply", "SimpleSmsReplyMostRecent", "SimpleSmsResend",
    "SimpleSmsSend", "SimpleSmsSendClipboardContent", "SimpleSmsSendReceivedAddress",
    "SportsTrackerActivitiesCountForWeek", "SportsTrackerActivitiesOnDate",
    "SportsTrackerActivityDuration", "SportsTrackerLongestDistanceActivity",
    "SportsTrackerTotalDistanceForCategoryOverInterval",
    "SportsTrackerTotalDurationForCategoryThisWeek",
    "SystemBluetoothTurnOff", "SystemBluetoothTurnOffVerify",
    "SystemBluetoothTurnOn", "SystemBluetoothTurnOnVerify",
    "SystemBrightnessMax", "SystemBrightnessMaxVerify",
    "SystemBrightnessMin", "SystemBrightnessMinVerify",
    "SystemCopyToClipboard",
    "SystemWifiTurnOff", "SystemWifiTurnOffVerify",
    "SystemWifiTurnOn", "SystemWifiTurnOnVerify",
    "TasksCompletedTasksForDate", "TasksDueNextWeek", "TasksDueOnDate",
    "TasksHighPriorityTasks", "TasksHighPriorityTasksDueOnDate",
    "TasksIncompleteTasksOnDate",
    "TurnOffWifiAndTurnOnBluetooth", "TurnOnWifiAndOpenApp",
    "VlcCreatePlaylist", "VlcCreateTwoPlaylists",
])
assert len(CANONICAL_TASKS) == 116, f"Expected 116 tasks, got {len(CANONICAL_TASKS)}"


# Per-task failures scraped from public agent benchmark pages.
DROIDRUN_FAILED = {
    "ContactsNewContactDraft", "MarkorTranscribeVideo", "OsmAndMarker",
    "RecipeAddMultipleRecipesFromImage", "RecipeAddMultipleRecipesFromMarkor2",
    "RecipeDeleteDuplicateRecipes3", "RetroPlaylistDuration",
    "TasksCompletedTasksForDate", "TasksIncompleteTasksOnDate",
    "BrowserDraw",
}
FINALRUN_FAILED = {
    "BrowserMultiply", "ExpenseAddMultipleFromGallery",
    "ExpenseAddMultipleFromMarkor", "ExpenseDeleteDuplicates",
    "ExpenseDeleteDuplicates2", "MarkorAddNoteHeader",
    "MarkorCreateNoteFromClipboard", "MarkorEditNote", "MarkorMergeNotes",
    "MarkorTranscribeVideo", "OsmAndMarker", "OsmAndTrack",
    "RecipeAddMultipleRecipesFromMarkor2", "RecipeDeleteDuplicateRecipes2",
    "RecipeDeleteDuplicateRecipes3", "RecipeDeleteMultipleRecipesWithConstraint",
    "RetroPlaylistDuration", "SimpleCalendarDeleteEventsOnRelativeDay",
    "SimpleCalendarEventsInNextWeek", "SimpleCalendarEventsInTimeRange",
    "SystemBrightnessMax", "SystemBrightnessMaxVerify",
    "TasksCompletedTasksForDate", "TasksDueNextWeek", "TasksHighPriorityTasks",
    "VlcCreatePlaylist", "VlcCreateTwoPlaylists",
}
AUTODEVICE_FAILED = {
    "MarkorAddNoteHeader", "MarkorChangeNoteContent", "MarkorMergeNotes",
    "MarkorTranscribeVideo", "RecipeDeleteDuplicateRecipes2",
    "RecipeDeleteDuplicateRecipes3",
}

PERTASK_AGENTS = {
    "DroidRun":   {"failed": DROIDRUN_FAILED,   "model": "GPT-5 + Gemini 2.5 Pro"},
    "FinalRun":   {"failed": FINALRUN_FAILED,   "model": "GPT-5"},
    "AutoDevice": {"failed": AUTODEVICE_FAILED, "model": "Gemini 3 Pro + Sonnet 4.5"},
}


def build_long_form():
    bench_id = get_benchmark_id(
        "androidworld",
        name="AndroidWorld",
        license=INFO.get("license"),
        source_url=INFO.get("data_source_url"),
        description=INFO.get("description"),
    )

    rows = []
    for agent, info in PERTASK_AGENTS.items():
        subj = resolve_subject(agent)
        failed = info["failed"]
        for task in CANONICAL_TASKS:
            item = register_item(
                benchmark_id=bench_id,
                raw_item_id=task,
                content=f"AndroidWorld task: {task}",
            )
            rows.append({
                "subject_id": subj,
                "item_id": item,
                "benchmark_id": bench_id,
                "trial": 1,
                "test_condition": None,
                "response": 0.0 if task in failed else 1.0,
                "correct_answer": None,
                "trace": None,
            })

    df = pd.DataFrame(rows, columns=[
        "subject_id", "item_id", "benchmark_id", "trial",
        "test_condition", "response", "correct_answer", "trace",
    ])
    df = ensure_unique_trials(df)
    df.to_parquet(RESPONSES_PATH, index=False)
    registry_save(CONTRIB_DIR)
    print(f"\n  wrote {RESPONSES_PATH.name} ({len(df):,} rows)")
    print(f"  wrote {CONTRIB_DIR.name}/{{subjects,items,benchmarks}}.parquet")
    return df


def print_stats(df: pd.DataFrame) -> None:
    if df.empty:
        print("  (empty DataFrame)")
        return
    print(f"\n  subjects: {df['subject_id'].nunique()}")
    print(f"  items:    {df['item_id'].nunique()}")
    print(f"  rows:     {len(df):,}")
    print(f"  max trial: {df['trial'].max()}")
    print(f"  response mean: {df['response'].mean():.3f}")


def main():
    print(f"[androidworld] building from {_BENCHMARK_DIR}")
    df = build_long_form()
    print_stats(df)


if __name__ == "__main__":
    main()
