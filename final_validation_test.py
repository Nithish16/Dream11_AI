#!/usr/bin/env python3
import asyncio
import sys

try:
    from tests.final_validation_test import main
except Exception as e:
    print(f"❌ Failed to import tests.final_validation_test: {e}")
    sys.exit(1)

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)
