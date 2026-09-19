"""Run the whole suite against the installed package instead of `src/`.

    MECHANISM_SELECTOR_TEST_INSTALLED=1 pytest tests/

With the variable set, the modules the tests import as `accounting`,
`mechanisms` and `selector` are the ones from the installed `mechanism_selector`
package, so every existing test exercises the published code. Without it, this
file does nothing.
"""

import os
import sys

if os.environ.get("MECHANISM_SELECTOR_TEST_INSTALLED") == "1":
    import mechanism_selector.accounting as _accounting
    import mechanism_selector.mechanisms as _mechanisms
    import mechanism_selector.selector as _selector

    sys.modules["accounting"] = _accounting
    sys.modules["mechanisms"] = _mechanisms
    sys.modules["selector"] = _selector
    print(f"\n[conftest] testing installed package at {os.path.dirname(_selector.__file__)}")
