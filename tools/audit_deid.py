"""Tutorial-compatible count-only audit entry point. Run in a separate Terminal."""
import sys
from deidentify_audit import main

if __name__ == '__main__':
    sys.argv.insert(1, 'audit')
    raise SystemExit(main())
