import re
import sys
from datetime import datetime, timedelta, timezone

import boto3

STACK = "agentic-rag"
REPORT = re.compile(
    r"REPORT RequestId: (\S+)\s+Duration: ([\d.]+) ms\s+Billed Duration: ([\d.]+) ms\s+"
    r"Memory Size: (\d+) MB\s+Max Memory Used: (\d+) MB(?:\s+Init Duration: ([\d.]+) ms)?"
)


def main() -> None:
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 6
    region = boto3.session.Session().region_name or "ap-south-1"

    stack = boto3.client("cloudformation", region_name=region).describe_stacks(StackName=STACK)["Stacks"][0]
    function = {o["OutputKey"]: o["OutputValue"] for o in stack["Outputs"]}["WorkerFunctionName"]
    group = boto3.client("lambda", region_name=region).get_function_configuration(
        FunctionName=function)["LoggingConfig"]["LogGroup"]

    logs = boto3.client("logs", region_name=region)
    start = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp() * 1000)
    kwargs = dict(logGroupName=group, startTime=start, filterPattern='"REPORT RequestId"')

    rows = []
    while True:
        page = logs.filter_log_events(**kwargs)
        for event in page["events"]:
            m = REPORT.search(event["message"])
            if m:
                status = re.search(r"Status: (\w+)", event["message"])
                rows.append((event["timestamp"], m, status.group(1) if status else "ok"))
        if not page.get("nextToken"):
            break
        kwargs["nextToken"] = page["nextToken"]

    if not rows:
        print(f"No finished runs in the last {hours:g} hours.")
        return

    print(f"{'time (UTC)':<9} {'run s':>7} {'memory used':>14} {'cold start s':>13} {'GB-s':>6}  status")
    peak = 0
    for ts, m, status in sorted(rows, key=lambda r: r[0]):
        _, duration, billed, size, used, init = m.groups()
        peak = max(peak, int(used))
        gb_s = int(size) / 1024 * float(billed) / 1000
        when = datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%H:%M:%S")
        print(f"{when:<9} {float(duration) / 1000:>7.1f} {used + ' / ' + size + ' MB':>14} "
              f"{(float(init) / 1000 if init else 0):>13.1f} {gb_s:>6.0f}  {status}")
    print(f"\nPeak memory in this window: {peak} MB (limit {rows[-1][1].group(4)} MB)")


if __name__ == "__main__":
    main()