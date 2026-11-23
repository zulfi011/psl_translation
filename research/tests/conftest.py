import os
import tempfile

# isolate every test run from the real data/ and experiments/ folders
os.environ.setdefault("PSL_RESEARCH_ROOT", tempfile.mkdtemp(prefix="psl_research_test_"))
