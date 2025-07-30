"""
Task Constants

File này định nghĩa các hằng số được sử dụng trong các worker tasks,
bao gồm tên các queue, thời gian timeout, và các constant khác.
"""

# Queue names
DOCUMENT_QUEUE = "document_jobs"

# Timeout settings (seconds)
QUEUE_POLL_TIMEOUT = 3
WORKER_SLEEP_INTERVAL = 1
