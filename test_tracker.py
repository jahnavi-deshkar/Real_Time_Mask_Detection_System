from src.tracker import ViolationTracker

tracker = ViolationTracker()

print("Testing violation logger...")

result1 = tracker.log_violation(2)
print("First log:", result1)

result2 = tracker.log_violation(3)
print("Second log:", result2)

print("Tracker test complete.")