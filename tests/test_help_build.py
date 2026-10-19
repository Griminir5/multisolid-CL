"""Help stays reproducible through Git checkouts; stale content is still rejected."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tools import build_help


class HelpBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = build_help.build_pages()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="multisolid-help-check-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.destination = self.root / "help"
        self.destination.mkdir()
        for name, text in self.pages.items():
            (self.destination / name).write_bytes(text.encode("utf-8"))

    def test_git_checkouts_preserve_generated_help(self):
        repository = self.root / "repository"
        repository.mkdir()
        relative = build_help.DESTINATION.relative_to(build_help.ROOT)
        shutil.copytree(self.destination, repository / relative)
        shutil.copy2(build_help.ROOT / ".gitattributes", repository / ".gitattributes")
        # This control file confirms that the test actually enables Windows
        # conversion, while the generated help is protected by its attributes.
        (repository / "ordinary.txt").write_bytes(b"first\nsecond\n")

        def git(*args, autocrlf="false"):
            subprocess.run(["git", "-C", str(repository), "-c", f"core.autocrlf={autocrlf}",
                            "-c", "core.attributesFile=", *args],
                           check=True, capture_output=True, text=True, timeout=30)

        git("init", "--quiet")
        git("add", ".")
        for autocrlf, newline in (("false", b"\n"), ("true", b"\r\n")):
            with self.subTest(autocrlf=autocrlf):
                checkout = self.root / f"checkout-{autocrlf}"
                git("checkout-index", "--all", "--prefix=" + checkout.as_posix() + "/", autocrlf=autocrlf)
                self.assertEqual((checkout / "ordinary.txt").read_bytes(),
                                 b"first" + newline + b"second" + newline)
                build_help.build(checkout / relative, check=True)
                for name, text in self.pages.items():
                    self.assertEqual((checkout / relative / name).read_bytes(), text.encode("utf-8"))

    def test_changed_content_is_rejected(self):
        path = self.destination / "index.html"
        path.write_bytes(path.read_bytes() + b"<!-- stale content -->\n")
        with self.assertRaisesRegex(ValueError, "Bundled help is stale.*index.html"):
            build_help.build(self.destination, check=True)

    def test_missing_page_is_rejected(self):
        (self.destination / "index.html").unlink()
        with self.assertRaisesRegex(ValueError, "Bundled help is stale.*index.html"):
            build_help.build(self.destination, check=True)

    def test_obsolete_page_is_rejected(self):
        (self.destination / "obsolete.html").write_bytes(b"outdated page\n")
        with self.assertRaisesRegex(ValueError, "Bundled help is stale.*obsolete.html"):
            build_help.build(self.destination, check=True)


if __name__ == "__main__":
    unittest.main()
