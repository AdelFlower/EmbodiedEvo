"""CPU regression tests for saved completion and process shutdown ordering."""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from AgenticSimulator.orchestrator import __main__ as cli
from AgenticSimulator.orchestrator import session, session_runner
from AgenticSimulator.orchestrator.files import write_json


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "202611130000_test"
        self.run.mkdir()
        self.read_fd, self.write_fd = os.pipe()
        self.addCleanup(os.close, self.read_fd)
        self.addCleanup(os.close, self.write_fd)

    def save_finished(self, **overrides):
        write_json(self.run / "session-result.json", {
            "closed": True,
            "reason": "agent_finished",
            "active_request": None,
            **overrides,
        })

    def event(self, kind):
        os.write(self.write_fd, (json.dumps({"event": kind}) + "\n").encode())

    def supervise(self, service_rc=None, agent=None):
        if agent is None:
            agent = Mock(returncode=0)
            agent.poll.return_value = 0
        service = Mock()
        service.poll.return_value = service_rc
        return session_runner.supervise(
            service, self.read_fd, lambda: agent, run=self.run,
            startup_timeout=1, timeout=1, event_log=io.StringIO(),
        )[0]

    def test_saved_finish_succeeds_before_finished_notification(self):
        for service_rc in (None, 0):
            with self.subTest(service_rc=service_rc):
                self.save_finished()
                self.event("ready")
                self.assertEqual(self.supervise(service_rc), "session_finished")

    def test_finished_notification_cannot_replace_saved_result(self):
        self.event("ready")
        self.event("finished")
        self.assertEqual(self.supervise(), "agent_exited")

    def test_incomplete_or_failed_state_is_not_completion(self):
        for state in (
            {"closed": False},
            {"reason": "executing", "active_request": "000001"},
            {"active_request": "000001"},
            {"reason": "cleanup_error"},
        ):
            with self.subTest(state=state):
                self.save_finished(**state)
                self.event("ready")
                self.assertEqual(self.supervise(), "agent_exited")

    def test_actual_service_failure_is_not_hidden_by_saved_finish(self):
        self.save_finished()
        self.event("ready")
        self.assertEqual(self.supervise(service_rc=1), "session_exited")

    def test_agent_failure_is_not_hidden_by_saved_finish(self):
        self.save_finished()
        self.event("ready")
        agent = Mock(returncode=1)
        agent.poll.return_value = 1
        self.assertEqual(self.supervise(agent=agent), "agent_exited")

    def test_agent_exit_still_waits_for_active_experiment(self):
        self.event("ready")
        os.write(self.write_fd, b'{"event":"executing","timeout":1}\n')
        agent = Mock(returncode=0)

        def poll():
            # Finish the submitted work after the first read of the event pipe.
            if agent.poll.call_count == 1:
                self.save_finished()
                self.event("idle")
            return 0

        agent.poll.side_effect = poll
        self.assertEqual(self.supervise(agent=agent), "session_finished")
        self.assertEqual(agent.poll.call_count, 2)

    def test_runner_shutdown_does_not_make_cli_report_failure(self):
        template = self.root / "template"
        source = self.root / "source"
        template.mkdir()
        source.mkdir()
        cfg = {
            "workspace_template": str(template), "source_root": str(source),
            "agent": {"image": "test-image", "network": "none"},
            "session": {"python": "/unused/python"},
        }
        service = Mock(returncode=None)
        service.poll.side_effect = lambda: service.returncode
        agent = Mock(returncode=0)
        agent.poll.side_effect = lambda: agent.returncode

        def popen(argv, **kwargs):
            if "pass_fds" in kwargs:
                self.save_finished()
                os.write(kwargs["pass_fds"][0], b'{"event":"ready"}\n')
                return service
            return agent

        def stop(process):
            if process is service:
                service.returncode = -15

        self.run.rmdir()  # run_session creates a new run directory.
        with (
            patch.object(session_runner.launch_agent, "require_image"),
            patch.object(session_runner.launch_agent, "stage_home"),
            patch.object(session_runner, "agent_command", return_value=("test", ["agent"])),
            patch.object(session_runner.subprocess, "Popen", side_effect=popen),
            patch.object(session_runner.subprocess, "run"),
            patch.object(session_runner, "stop_process", side_effect=stop),
        ):
            result = session_runner.run_session(cfg, self.run)

        self.assertEqual(result["reason"], "session_finished")
        self.assertEqual(result["session_returncode"], -15)
        saved = json.loads((self.run / "runtime-result.json").read_text())
        self.assertEqual(saved, result)
        with (
            patch.object(cli, "load_config", return_value=cfg),
            patch.object(cli, "run_session", return_value=result),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            self.assertEqual(cli.main([
                "--config", "unused.json", "--run-dir", str(self.run),
            ]), 0)

    def test_shutdown_after_finish_preserves_saved_terminal_state(self):
        cfg = {"session": {"execution_timeout_sec": 1}}
        executor = Mock()
        with (
            patch.object(session.socket, "socket") as socket_factory,
            patch.object(session.os, "chmod"),
            patch.object(session, "receive", return_value={"op": "finish"}),
            # Emulate SIGTERM after finish was saved, before cleanup completed.
            patch.object(session, "send", side_effect=KeyboardInterrupt),
        ):
            listener = socket_factory.return_value.__enter__.return_value
            # Connections and their streams are context managers.
            listener.accept.return_value = (MagicMock(), None)
            session.serve(cfg, self.run, self.root / "session.sock", self.write_fd, executor)

        state = json.loads((self.run / "session-result.json").read_text())
        self.assertTrue(state["closed"])
        self.assertEqual(state["reason"], "agent_finished")
        executor.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
