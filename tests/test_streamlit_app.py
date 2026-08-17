"""Regression tests for the Streamlit wrapper."""

from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
from pathlib import Path
from unittest import mock

import pandas as pd


def _run_electoral_action(selected_state: str) -> mock.Mock:
    streamlit = mock.MagicMock()
    app_path = Path(__file__).parents[1] / "streamlit" / "streamlit_app.py"
    spec = spec_from_file_location("naampy_streamlit_app", app_path)
    assert spec is not None
    assert spec.loader is not None
    app_module = module_from_spec(spec)
    with mock.patch.dict("sys.modules", {"streamlit": streamlit}):
        spec.loader.exec_module(app_module)

    action = "Append Electoral Roll Data to First Name"
    streamlit.sidebar.selectbox.return_value = action
    streamlit.file_uploader.return_value = StringIO("name\nPriya\n")
    streamlit.selectbox.side_effect = ["name", selected_state]
    streamlit.button.return_value = True
    prediction = pd.DataFrame({"name": ["Priya"]})
    function = mock.Mock(return_value=prediction)
    app_module.sidebar_options[action] = function
    app_module.download_file = mock.Mock()

    app_module.app()

    function.assert_called_once()
    return function


def test_all_states_selection_uses_national_lookup() -> None:
    """The default UI choice must map to the core API's ``None`` sentinel."""
    function = _run_electoral_action("all")
    assert function.call_args.kwargs["state"] is None


def test_maharashtra_selection_uses_published_dataset_key() -> None:
    """The correctly spelled UI label must map to the source table's legacy key."""
    function = _run_electoral_action("maharashtra")
    assert function.call_args.kwargs["state"] == "maharastra"
