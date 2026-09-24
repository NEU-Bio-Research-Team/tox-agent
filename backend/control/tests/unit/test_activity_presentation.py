from toxagent.activities.presentation import activity_for_tool


def test_activity_label_uses_intent_not_raw_tool_name_alone():
    activity = activity_for_tool("get_analysis_slice", status="started", intent="attribution")
    assert activity["label_key"] == "activity.inspecting_factors"
    assert activity["visibility"] == "user"


def test_activity_falls_back_to_the_safe_generic_mapping():
    activity = activity_for_tool("unknown_tool", status="started", intent="report_qa")
    assert activity["label_key"] == "activity.processing"
