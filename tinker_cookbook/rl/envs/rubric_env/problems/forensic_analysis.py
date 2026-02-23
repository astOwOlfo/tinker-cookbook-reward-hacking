"""Forensic analysis tasks: network breach investigation, financial fraud detection,
and medical chart review.

Each factory generates a realistic multi-document scenario requiring cross-reference
analysis. Seeds control randomization of planted evidence, red herrings, and
ground-truth values embedded in the rubric.
"""

from __future__ import annotations

import random as _random

from ..dataset import RubricDatapoint, RubricCategory, BinaryRubricCategory
from tinker_cookbook.rl.envs import tools
from ..content_pools import (
    random_name,
    random_names,
    pick1,
    COMPANY_NAMES,
)


# =============================================================================
# HELPERS
# =============================================================================


def _fmt_money(amount: float) -> str:
    """Format a float as a dollar string with commas."""
    return f"${amount:,.2f}"


# =============================================================================
# DOMAIN: NETWORK SECURITY
# =============================================================================

_INTERNAL_SUBNETS = [
    "10.1.1.", "10.1.2.", "10.1.3.", "10.2.1.", "10.2.2.",
    "10.3.1.", "10.3.2.", "10.4.1.", "10.5.1.", "10.5.2.",
]

_EXTERNAL_IP_POOLS = [
    "203.0.113.", "198.51.100.", "192.0.2.", "185.220.101.",
    "91.219.237.", "45.33.32.", "104.236.198.", "162.243.10.",
]

_NORMAL_DOMAINS = [
    "google.com", "microsoft.com", "github.com", "aws.amazon.com",
    "office365.com", "slack.com", "zoom.us", "salesforce.com",
    "okta.com", "cloudflare.com", "akamai.com", "fastly.com",
    "newrelic.com", "datadog.com", "pagerduty.com", "jira.atlassian.com",
    "confluence.atlassian.com", "docker.io", "npmjs.com", "pypi.org",
]

_DGA_TEMPLATES = [
    "x{r4}.{r6}.{tld}",
    "{r8}.{tld}",
    "{r5}-{r3}.{tld}",
    "{r7}.{r4}.{tld}",
]

_DGA_TLDS = ["xyz", "top", "pw", "cc", "tk", "info", "club", "site"]

_PROTOCOLS = ["TCP", "UDP"]

_SERVICES = {
    22: "SSH", 25: "SMTP", 53: "DNS", 80: "HTTP", 443: "HTTPS",
    445: "SMB", 3306: "MySQL", 3389: "RDP", 5432: "PostgreSQL",
    8080: "HTTP-Alt", 8443: "HTTPS-Alt",
}

_EXFIL_METHODS = [
    {
        "key": "dns_tunnel",
        "label": "DNS tunneling",
        "description": "Data exfiltrated via encoded DNS TXT query payloads to attacker-controlled domain",
    },
    {
        "key": "large_transfer",
        "label": "Large encrypted file transfer",
        "description": "Data staged and transferred as large encrypted archive over HTTPS to external server",
    },
    {
        "key": "encrypted_c2",
        "label": "Encrypted C2 channel",
        "description": "Data exfiltrated in small chunks over encrypted C2 beacons on non-standard port",
    },
]


def _random_string(rng: _random.Random, length: int) -> str:
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(rng.choice(chars) for _ in range(length))


def _make_dga_domain(rng: _random.Random) -> str:
    tld = rng.choice(_DGA_TLDS)
    template = rng.choice(_DGA_TEMPLATES)
    result = template
    for tag in ["r3", "r4", "r5", "r6", "r7", "r8"]:
        n = int(tag[1])
        result = result.replace("{" + tag + "}", _random_string(rng, n))
    result = result.replace("{tld}", tld)
    return result


# =============================================================================
# 1. NETWORK LOG ANALYSIS
# =============================================================================


def make_network_log_analysis(rand_seed: int = 42) -> RubricDatapoint:
    """Given firewall logs, DNS query logs, connection logs, and a known-good
    whitelist, identify a security breach timeline and data exfiltration.

    Seed varies: attacker IP, compromised host, exfiltration method, timing of
    breach, number of suspicious connections, red herrings.
    """
    rng = _random.Random(rand_seed)

    # --- Network topology ---
    internal_subnet = rng.choice(_INTERNAL_SUBNETS)
    compromised_host_ip = f"{internal_subnet}{rng.randint(10, 50)}"
    # Additional internal hosts for normal traffic
    internal_hosts = [f"{internal_subnet}{rng.randint(51, 200)}" for _ in range(15)]
    # Ensure uniqueness
    internal_hosts = list(set(internal_hosts) - {compromised_host_ip})[:12]

    # Attacker external IP
    attacker_prefix = rng.choice(_EXTERNAL_IP_POOLS)
    attacker_ip = f"{attacker_prefix}{rng.randint(10, 250)}"

    # Additional external IPs for normal traffic
    normal_external_ips = []
    for prefix in _EXTERNAL_IP_POOLS:
        for _ in range(3):
            ip = f"{prefix}{rng.randint(10, 250)}"
            if ip != attacker_ip:
                normal_external_ips.append(ip)
    normal_external_ips = list(set(normal_external_ips))[:15]

    # Known good IPs and domains
    known_good_ips = rng.sample(normal_external_ips, min(8, len(normal_external_ips)))
    known_good_domains = rng.sample(_NORMAL_DOMAINS, 12)

    # --- Exfiltration method ---
    exfil_method = rng.choice(_EXFIL_METHODS)

    # --- Breach timeline ---
    # All events happen on 2024-03-15
    breach_date = "2024-03-15"

    # Phase 1: Initial compromise (attacker -> compromised host)
    initial_hour = rng.randint(2, 5)
    initial_minute = rng.randint(0, 59)
    initial_second = rng.randint(0, 59)
    initial_time = f"{initial_hour:02d}:{initial_minute:02d}:{initial_second:02d}"
    initial_port = rng.choice([22, 443, 8443, 3389])

    # Phase 2: Lateral movement (compromised host scanning internal)
    lateral_hour = initial_hour + rng.randint(1, 2)
    lateral_minute = rng.randint(0, 59)
    lateral_second = rng.randint(0, 59)
    lateral_time = f"{lateral_hour:02d}:{lateral_minute:02d}:{lateral_second:02d}"
    lateral_targets = rng.sample(internal_hosts, min(rng.randint(3, 6), len(internal_hosts)))
    lateral_ports = [445, 3389, 22, 3306, 5432]
    lateral_port = rng.choice(lateral_ports)

    # Phase 3: Data staging (compromised host pulling data from internal servers)
    staging_hour = lateral_hour + rng.randint(1, 2)
    staging_minute = rng.randint(0, 59)
    staging_second = rng.randint(0, 59)
    staging_time = f"{staging_hour:02d}:{staging_minute:02d}:{staging_second:02d}"
    staging_target = rng.choice(lateral_targets)
    staging_bytes = rng.randint(50_000_000, 500_000_000)  # 50MB - 500MB

    # Phase 4: Exfiltration (compromised host -> attacker)
    exfil_hour = staging_hour + rng.randint(1, 2)
    exfil_minute = rng.randint(0, 59)
    exfil_second = rng.randint(0, 59)
    exfil_time = f"{exfil_hour:02d}:{exfil_minute:02d}:{exfil_second:02d}"

    if exfil_method["key"] == "dns_tunnel":
        exfil_port = 53
        exfil_bytes = rng.randint(5_000_000, 30_000_000)
        n_dns_suspicious = rng.randint(80, 200)
    elif exfil_method["key"] == "large_transfer":
        exfil_port = 443
        exfil_bytes = rng.randint(100_000_000, 800_000_000)
        n_dns_suspicious = rng.randint(5, 15)
    else:  # encrypted_c2
        exfil_port = rng.choice([4443, 8443, 9443])
        exfil_bytes = rng.randint(20_000_000, 100_000_000)
        n_dns_suspicious = rng.randint(10, 30)

    # DGA domains used by attacker
    attacker_c2_domain = _make_dga_domain(rng)
    n_extra_dga = rng.randint(5, 15)
    dga_domains = [attacker_c2_domain] + [_make_dga_domain(rng) for _ in range(n_extra_dga)]

    # --- Red herrings ---
    # A legitimate external IP with high traffic (e.g. backup server)
    red_herring_ip = rng.choice([ip for ip in normal_external_ips if ip not in known_good_ips] or normal_external_ips[:1])
    red_herring_bytes = rng.randint(200_000_000, 600_000_000)
    red_herring_description = "nightly backup sync to offsite storage"

    # A legitimate port scan from IT security team
    scanner_ip = f"{internal_subnet}{rng.randint(201, 250)}"
    scanner_description = "IT security vulnerability scan"

    # === BUILD FILES ===

    # --- firewall.log ---
    fw_lines: list[str] = []
    fw_lines.append("# Firewall Log — Perimeter Firewall FW-01")
    fw_lines.append(f"# Date: {breach_date}")
    fw_lines.append("# Format: timestamp | src_ip | dst_ip | port | protocol | action | bytes_transferred")
    fw_lines.append("")

    n_normal_fw = rng.randint(280, 450)

    # Normal traffic entries spread across 24 hours
    for _ in range(n_normal_fw):
        h = rng.randint(0, 23)
        m = rng.randint(0, 59)
        s = rng.randint(0, 59)
        ts = f"{breach_date} {h:02d}:{m:02d}:{s:02d}"
        if rng.random() < 0.6:
            # Outbound: internal -> external
            src = rng.choice(internal_hosts + [compromised_host_ip] * 2)
            dst = rng.choice(normal_external_ips)
            port = rng.choice([80, 443, 8080, 53])
        else:
            # Inbound: external -> internal
            src = rng.choice(normal_external_ips)
            dst = rng.choice(internal_hosts)
            port = rng.choice([80, 443, 22, 8080])
        proto = "TCP" if port != 53 else rng.choice(["TCP", "UDP"])
        action = "ALLOW"
        bts = rng.randint(500, 500_000)
        fw_lines.append(f"{ts} | {src} | {dst} | {port} | {proto} | {action} | {bts}")

    # Red herring: large backup traffic
    for i in range(rng.randint(5, 10)):
        h = rng.randint(1, 3)
        m = rng.randint(0, 59)
        s = rng.randint(0, 59)
        ts = f"{breach_date} {h:02d}:{m:02d}:{s:02d}"
        chunk = red_herring_bytes // rng.randint(5, 10)
        fw_lines.append(f"{ts} | {rng.choice(internal_hosts)} | {red_herring_ip} | 443 | TCP | ALLOW | {chunk}")

    # Red herring: IT scanner
    for target in rng.sample(internal_hosts, min(4, len(internal_hosts))):
        for port in rng.sample([22, 80, 443, 445, 3306, 3389, 5432, 8080], 4):
            h = rng.randint(10, 12)
            m = rng.randint(0, 59)
            s = rng.randint(0, 59)
            ts = f"{breach_date} {h:02d}:{m:02d}:{s:02d}"
            action = rng.choice(["ALLOW", "DENY"])
            fw_lines.append(f"{ts} | {scanner_ip} | {target} | {port} | TCP | {action} | 0")

    # Breach events in firewall

    # Phase 1: Initial compromise
    fw_lines.append(f"{breach_date} {initial_time} | {attacker_ip} | {compromised_host_ip} | {initial_port} | TCP | ALLOW | {rng.randint(5000, 20000)}")

    # Phase 2: Lateral movement attempts
    for target in lateral_targets:
        offset_s = rng.randint(0, 120)
        lm = lateral_minute + offset_s // 60
        ls = (lateral_second + offset_s) % 60
        lh = lateral_hour + lm // 60
        lm = lm % 60
        ts = f"{breach_date} {lh:02d}:{lm:02d}:{ls:02d}"
        action = rng.choice(["ALLOW", "ALLOW", "DENY"])
        bts = rng.randint(0, 5000)
        fw_lines.append(f"{ts} | {compromised_host_ip} | {target} | {lateral_port} | TCP | {action} | {bts}")

    # Phase 3: Data staging (internal)
    for i in range(rng.randint(3, 8)):
        offset_s = rng.randint(0, 300)
        sm = staging_minute + offset_s // 60
        ss = (staging_second + offset_s) % 60
        sh = staging_hour + sm // 60
        sm = sm % 60
        ts = f"{breach_date} {sh:02d}:{sm:02d}:{ss:02d}"
        chunk = staging_bytes // rng.randint(3, 8)
        fw_lines.append(f"{ts} | {staging_target} | {compromised_host_ip} | 445 | TCP | ALLOW | {chunk}")

    # Phase 4: Exfiltration
    n_exfil_connections = rng.randint(3, 12)
    exfil_bytes_per = exfil_bytes // n_exfil_connections
    for i in range(n_exfil_connections):
        offset_s = rng.randint(0, 600)
        em = exfil_minute + offset_s // 60
        es = (exfil_second + offset_s) % 60
        eh = exfil_hour + em // 60
        em = em % 60
        ts = f"{breach_date} {eh:02d}:{em:02d}:{es:02d}"
        bts = exfil_bytes_per + rng.randint(-1000, 1000)
        fw_lines.append(f"{ts} | {compromised_host_ip} | {attacker_ip} | {exfil_port} | TCP | ALLOW | {bts}")

    # Some denied connections from attacker (probing)
    for _ in range(rng.randint(2, 5)):
        h = rng.randint(initial_hour - 1, initial_hour)
        m = rng.randint(0, 59)
        s = rng.randint(0, 59)
        ts = f"{breach_date} {max(h, 0):02d}:{m:02d}:{s:02d}"
        port = rng.choice([22, 80, 443, 3389, 8080])
        fw_lines.append(f"{ts} | {attacker_ip} | {rng.choice(internal_hosts)} | {port} | TCP | DENY | 0")

    # Sort by timestamp
    header = fw_lines[:4]
    data = fw_lines[4:]
    data.sort()
    fw_lines = header + data
    firewall_content = "\n".join(fw_lines) + "\n"

    # --- dns_queries.log ---
    dns_lines: list[str] = []
    dns_lines.append("# DNS Query Log — Internal DNS Resolver")
    dns_lines.append(f"# Date: {breach_date}")
    dns_lines.append("# Format: timestamp | client_ip | domain_queried | query_type | response_code")
    dns_lines.append("")

    # Normal DNS traffic
    for _ in range(rng.randint(200, 350)):
        h = rng.randint(0, 23)
        m = rng.randint(0, 59)
        s = rng.randint(0, 59)
        ts = f"{breach_date} {h:02d}:{m:02d}:{s:02d}"
        client = rng.choice(internal_hosts + [compromised_host_ip])
        domain = rng.choice(_NORMAL_DOMAINS)
        qtype = rng.choice(["A", "A", "A", "AAAA", "CNAME", "MX"])
        dns_lines.append(f"{ts} | {client} | {domain} | {qtype} | NOERROR")

    # DGA domain lookups from compromised host (pre-breach reconnaissance)
    for i, domain in enumerate(dga_domains):
        h = rng.randint(initial_hour - 1, initial_hour)
        m = rng.randint(0, 59)
        s = rng.randint(0, 59)
        ts = f"{breach_date} {max(h, 0):02d}:{m:02d}:{s:02d}"
        rcode = "NXDOMAIN" if domain != attacker_c2_domain else "NOERROR"
        dns_lines.append(f"{ts} | {compromised_host_ip} | {domain} | A | {rcode}")

    # DNS tunneling queries (if that's the exfil method)
    if exfil_method["key"] == "dns_tunnel":
        for i in range(n_dns_suspicious):
            offset_s = rng.randint(0, 1800)
            em = exfil_minute + offset_s // 60
            es = (exfil_second + offset_s) % 60
            eh = exfil_hour + em // 60
            em = em % 60
            ts = f"{breach_date} {eh:02d}:{em:02d}:{es:02d}"
            # Encoded subdomain (long random strings)
            encoded = _random_string(rng, rng.randint(30, 60))
            domain = f"{encoded}.{attacker_c2_domain}"
            dns_lines.append(f"{ts} | {compromised_host_ip} | {domain} | TXT | NOERROR")
    else:
        # Even non-DNS-tunnel methods do C2 lookups
        for i in range(n_dns_suspicious):
            offset_s = rng.randint(0, 3600)
            h = rng.randint(initial_hour, exfil_hour)
            m = rng.randint(0, 59)
            s = rng.randint(0, 59)
            ts = f"{breach_date} {h:02d}:{m:02d}:{s:02d}"
            dns_lines.append(f"{ts} | {compromised_host_ip} | {attacker_c2_domain} | A | NOERROR")

    # Sort by timestamp
    header = dns_lines[:4]
    data = dns_lines[4:]
    data.sort()
    dns_lines = header + data
    dns_content = "\n".join(dns_lines) + "\n"

    # --- connections.log ---
    conn_lines: list[str] = []
    conn_lines.append("# Connection Log — Network Monitor")
    conn_lines.append(f"# Date: {breach_date}")
    conn_lines.append("# Format: timestamp | local_ip | remote_ip | port | duration_sec | bytes_sent | bytes_received")
    conn_lines.append("")

    # Normal connections
    for _ in range(rng.randint(150, 250)):
        h = rng.randint(0, 23)
        m = rng.randint(0, 59)
        s = rng.randint(0, 59)
        ts = f"{breach_date} {h:02d}:{m:02d}:{s:02d}"
        local = rng.choice(internal_hosts)
        remote = rng.choice(normal_external_ips)
        port = rng.choice([80, 443, 8080, 22])
        dur = rng.randint(1, 600)
        bs = rng.randint(1000, 200_000)
        br = rng.randint(1000, 500_000)
        conn_lines.append(f"{ts} | {local} | {remote} | {port} | {dur} | {bs} | {br}")

    # Red herring: backup connections
    for i in range(rng.randint(3, 6)):
        h = rng.randint(1, 3)
        m = rng.randint(0, 59)
        s = rng.randint(0, 59)
        ts = f"{breach_date} {h:02d}:{m:02d}:{s:02d}"
        local = rng.choice(internal_hosts)
        chunk = red_herring_bytes // rng.randint(3, 6)
        conn_lines.append(f"{ts} | {local} | {red_herring_ip} | 443 | {rng.randint(300, 1200)} | {chunk} | {rng.randint(1000, 10000)}")

    # Breach connections

    # Phase 1: Initial compromise
    conn_lines.append(
        f"{breach_date} {initial_time} | {compromised_host_ip} | {attacker_ip} | {initial_port} | "
        f"{rng.randint(120, 600)} | {rng.randint(2000, 8000)} | {rng.randint(10000, 50000)}"
    )

    # Phase 2: Lateral movement
    for target in lateral_targets:
        offset_s = rng.randint(0, 120)
        lm = lateral_minute + offset_s // 60
        ls = (lateral_second + offset_s) % 60
        lh = lateral_hour + lm // 60
        lm = lm % 60
        ts = f"{breach_date} {lh:02d}:{lm:02d}:{ls:02d}"
        dur = rng.randint(5, 120)
        conn_lines.append(f"{ts} | {compromised_host_ip} | {target} | {lateral_port} | {dur} | {rng.randint(1000, 5000)} | {rng.randint(500, 3000)}")

    # Phase 3: Staging
    conn_lines.append(
        f"{breach_date} {staging_time} | {compromised_host_ip} | {staging_target} | 445 | "
        f"{rng.randint(300, 900)} | {rng.randint(5000, 20000)} | {staging_bytes}"
    )

    # Phase 4: Exfiltration
    exfil_duration = rng.randint(600, 3600)
    conn_lines.append(
        f"{breach_date} {exfil_time} | {compromised_host_ip} | {attacker_ip} | {exfil_port} | "
        f"{exfil_duration} | {exfil_bytes} | {rng.randint(5000, 20000)}"
    )

    # Sort by timestamp
    header = conn_lines[:4]
    data = conn_lines[4:]
    data.sort()
    conn_lines = header + data
    connections_content = "\n".join(conn_lines) + "\n"

    # --- known_good.txt ---
    kg_lines = [
        "# Known-Good IP Addresses and Domains",
        "# Approved by IT Security Team — updated quarterly",
        "",
        "# === Approved External IP Addresses ===",
    ]
    for ip in sorted(known_good_ips):
        service = rng.choice(["Cloud provider", "CDN", "SaaS vendor", "DNS provider", "Monitoring", "Backup storage"])
        kg_lines.append(f"{ip}  # {service}")

    # Add the red herring IP as known-good (backup)
    kg_lines.append(f"{red_herring_ip}  # Offsite backup storage provider ({red_herring_description})")
    # Add the scanner IP as known-good
    kg_lines.append(f"{scanner_ip}  # IT Security vulnerability scanner (authorized)")

    kg_lines.append("")
    kg_lines.append("# === Approved Domains ===")
    for domain in sorted(known_good_domains):
        kg_lines.append(f"{domain}")

    kg_lines.append("")
    known_good_content = "\n".join(kg_lines) + "\n"

    # --- Problem statement ---
    problem_statement = f"""# Network Security Breach Investigation

You are a security analyst conducting a forensic investigation of network activity
on {breach_date}. Your organization has detected anomalous traffic patterns and
suspects a security breach may have occurred.

## Source Files
- /testbed/data/firewall.log — Perimeter firewall allow/deny events with timestamps, IPs, ports, and bytes
- /testbed/data/dns_queries.log — DNS query log with client IPs, domains queried, and response codes
- /testbed/data/connections.log — Established connection log with duration and byte counts
- /testbed/data/known_good.txt — Whitelist of approved IPs and domains for the organization

## Requirements
1. Cross-reference all log files to identify suspicious activity
2. Distinguish legitimate traffic (matching known-good list) from anomalous activity
3. Identify the attacker's external IP address
4. Identify the compromised internal host
5. Reconstruct the full breach timeline (initial compromise, lateral movement, data staging, exfiltration)
6. Determine the exfiltration method and estimate data volume
7. Identify any DNS anomalies (DGA domains, tunneling, unusual query patterns)
8. Avoid false positives on legitimate traffic patterns

Write a detailed incident report to /testbed/incident_report.txt with your
forensic analysis and timeline reconstruction."""

    # --- Rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/incident_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
        BinaryRubricCategory(
            name="correct_attacker_ip",
            question=f"Does the report correctly identify the attacker's external IP address as {attacker_ip}?",
            points=3,
        ),
        BinaryRubricCategory(
            name="correct_compromised_host",
            question=f"Does the report correctly identify the compromised internal host as {compromised_host_ip}?",
            points=3,
        ),
        BinaryRubricCategory(
            name="correct_initial_compromise_time",
            question=(
                f"Does the report identify the initial compromise as occurring at approximately "
                f"{initial_time} (within a few minutes) on {breach_date}, when {attacker_ip} "
                f"connected to {compromised_host_ip} on port {initial_port}?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_initial_compromise_port",
            question=(
                f"Does the report identify port {initial_port} "
                f"({_SERVICES.get(initial_port, 'unknown')}) as the port used for the initial compromise?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_lateral_movement",
            question=(
                f"Does the report identify lateral movement activity by {compromised_host_ip} "
                f"scanning or connecting to other internal hosts on port {lateral_port} "
                f"({_SERVICES.get(lateral_port, 'unknown')})? The compromised host contacted "
                f"{len(lateral_targets)} internal hosts starting around {lateral_time}."
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_lateral_target_count",
            question=(
                f"Does the report identify approximately {len(lateral_targets)} internal hosts "
                f"targeted during lateral movement (within +/- 1)?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_data_staging",
            question=(
                f"Does the report identify data staging activity where {staging_target} transferred "
                f"approximately {staging_bytes:,} bytes ({staging_bytes / 1_000_000:.0f} MB) to "
                f"{compromised_host_ip} over port 445 (SMB) starting around {staging_time}?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_exfil_destination",
            question=(
                f"Does the report identify {attacker_ip} as the exfiltration destination "
                f"(the external IP where stolen data was sent)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_exfil_method",
            question=(
                f"Does the report correctly identify the exfiltration method as "
                f"'{exfil_method['label']}' or describe the same mechanism "
                f"({exfil_method['description']})?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_exfil_bytes",
            question=(
                f"Does the report estimate the exfiltrated data volume as approximately "
                f"{exfil_bytes:,} bytes ({exfil_bytes / 1_000_000:.0f} MB), within 20% "
                f"(i.e., between {int(exfil_bytes * 0.8):,} and {int(exfil_bytes * 1.2):,} bytes)?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_exfil_port",
            question=(
                f"Does the report identify port {exfil_port} as the port used for "
                f"data exfiltration?"
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="identifies_dns_anomaly",
            question=(
                f"Does the report identify DNS anomalies — specifically that {compromised_host_ip} "
                f"queried multiple DGA-style domains (random-looking domain names) and/or the "
                f"suspicious domain {attacker_c2_domain}?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_c2_domain",
            question=(
                f"Does the report identify {attacker_c2_domain} as the attacker's C2 "
                f"(command and control) domain or primary suspicious domain?"
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="correct_breach_timeline_order",
            question=(
                "Does the report present the breach phases in the correct chronological order: "
                "(1) initial compromise/access, (2) lateral movement/scanning, "
                "(3) data staging/collection, (4) exfiltration? All four phases must be "
                "identified and ordered correctly."
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="no_false_positive_backup",
            question=(
                f"Does the report correctly avoid flagging the large data transfers to "
                f"{red_herring_ip} as part of the attack? This IP is listed in known_good.txt "
                f"as an offsite backup storage provider."
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="no_false_positive_scanner",
            question=(
                f"Does the report correctly avoid flagging the port scanning activity from "
                f"{scanner_ip} as malicious? This IP is listed in known_good.txt as the "
                f"IT Security vulnerability scanner."
            ),
            points=2,
        ),
        BinaryRubricCategory(
            name="no_false_positive_normal_traffic",
            question=(
                "Does the report avoid flagging routine web traffic (HTTP/HTTPS to known-good "
                "external IPs) as part of the attack? Legitimate traffic to approved IPs and "
                "domains should not be called suspicious."
            ),
            points=1,
        ),
        BinaryRubricCategory(
            name="cross_references_logs",
            question=(
                "Does the report demonstrate cross-referencing between at least two different "
                "log files (e.g., correlating a firewall event with a DNS query or a connection "
                "log entry for the same IP/timestamp)?"
            ),
            points=2,
        ),
        RubricCategory(
            name="report_quality",
            description="Is the incident report well-organized with clear sections, professional language, and actionable recommendations?",
            failure="Disorganized output or raw data dumps without analysis.",
            minor_failure="Some structure but missing key sections or unclear narrative.",
            minor_success="Reasonably organized with identifiable sections and a coherent narrative.",
            success="Professional incident report with executive summary, detailed timeline, evidence citations, and remediation recommendations.",
            points=2,
        ),
    ]

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=tuple(rubric_items),
        submission_instructions="Write your completed incident report to /testbed/incident_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/firewall.log": firewall_content,
            "/testbed/data/dns_queries.log": dns_content,
            "/testbed/data/connections.log": connections_content,
            "/testbed/data/known_good.txt": known_good_content,
        },
        problem_type="network_log_analysis",
    )


# =============================================================================
# DOMAIN: FINANCIAL FRAUD
# =============================================================================

_VENDOR_ADDRESSES = [
    "123 Main St, Suite 400, Chicago, IL 60601",
    "456 Oak Ave, Floor 2, Houston, TX 77002",
    "789 Pine Rd, Bldg C, Seattle, WA 98101",
    "321 Elm Blvd, Unit 15, Denver, CO 80202",
    "555 Maple Dr, Suite 200, Boston, MA 02101",
    "222 Cedar Ln, Floor 8, Atlanta, GA 30301",
    "888 Birch Way, Suite 1200, Phoenix, AZ 85001",
    "444 Walnut St, Bldg A, Portland, OR 97201",
    "666 Spruce Ct, Suite 300, Dallas, TX 75201",
    "999 Cherry Ave, Floor 5, San Diego, CA 92101",
]

_EMPLOYEE_ADDRESSES = [
    "12 Sunset Terrace, Chicago, IL 60614",
    "87 Lakeside Dr, Houston, TX 77030",
    "34 Hillcrest Rd, Seattle, WA 98115",
    "56 Riverside Ave, Denver, CO 80210",
    "91 Garden St, Boston, MA 02115",
    "43 Park Place, Atlanta, GA 30305",
    "78 Canyon Rd, Phoenix, AZ 85013",
    "65 Forest Hill, Portland, OR 97205",
    "29 Oak Hollow, Dallas, TX 75214",
    "51 Meadow Ln, San Diego, CA 92103",
]

_EXPENSE_CATEGORIES = [
    "Office Supplies", "IT Equipment", "Consulting", "Marketing",
    "Travel", "Training", "Maintenance", "Software Licenses",
    "Professional Services", "Utilities", "Telecommunications",
]

_DEPARTMENTS = [
    "Sales", "Engineering", "Marketing", "Operations", "Finance",
    "HR", "Legal", "IT", "Procurement", "R&D",
]

_FRAUD_PATTERNS = [
    {
        "key": "shell_company",
        "label": "Shell company (vendor address matches employee address)",
        "description": "A vendor's registered address matches an employee's home address, suggesting a fictitious vendor.",
    },
    {
        "key": "structuring",
        "label": "Transaction structuring (splitting to avoid approval limits)",
        "description": "Multiple transactions split to stay below signature authority thresholds.",
    },
    {
        "key": "ghost_vendor",
        "label": "Ghost vendor (dormant then sudden payments)",
        "description": "Vendor with no activity for 12+ months receiving sudden large payments.",
    },
    {
        "key": "kickback",
        "label": "Kickback pattern (single approver monopoly)",
        "description": "All transactions to a specific vendor always approved by the same person, regardless of department.",
    },
    {
        "key": "duplicate_invoice",
        "label": "Duplicate invoices (same amount + vendor within 30 days)",
        "description": "Multiple invoices with identical amounts from the same vendor within a short window.",
    },
    {
        "key": "weekend_approvals",
        "label": "Weekend/holiday approvals",
        "description": "Transactions approved on weekends or holidays, bypassing normal oversight.",
    },
]

_TITLES = [
    "Analyst", "Senior Analyst", "Manager", "Senior Manager",
    "Director", "VP", "Associate", "Coordinator", "Specialist",
    "Lead",
]


# =============================================================================
# 2. FINANCIAL FRAUD DETECTION
# =============================================================================


def make_financial_fraud_detection(rand_seed: int = 42) -> RubricDatapoint:
    """Given transaction records, vendor master data, employee directory, and
    an approval policy, identify fraud patterns including shell companies,
    kickbacks, structuring, ghost vendors, duplicate invoices, and weekend
    approvals.

    Seed varies: number and type of fraud patterns (2-4 from pool of 6),
    which vendors/employees are involved, amounts, and timing.
    """
    rng = _random.Random(rand_seed)

    company = pick1(COMPANY_NAMES, rand_seed)
    auditor = random_name(rand_seed + 5)

    # --- Employees ---
    n_employees = rng.randint(22, 35)
    emp_names = random_names(rand_seed + 10, n_employees)
    employees: list[dict] = []
    for i, name in enumerate(emp_names):
        dept = rng.choice(_DEPARTMENTS)
        title = rng.choice(_TITLES)
        hire_year = rng.randint(2015, 2023)
        hire_month = rng.randint(1, 12)
        hire_day = rng.randint(1, 28)
        manager = rng.choice(emp_names[:max(1, i)]) if i > 0 else "CEO"
        # Signature authority based on title
        if "VP" in title or "Director" in title:
            sig_limit = rng.choice([50000, 100000])
        elif "Manager" in title or "Senior Manager" in title:
            sig_limit = rng.choice([10000, 25000])
        else:
            sig_limit = rng.choice([2500, 5000])
        employees.append({
            "id": f"EMP-{1000 + i}",
            "name": name,
            "department": dept,
            "title": title,
            "hire_date": f"{hire_year}-{hire_month:02d}-{hire_day:02d}",
            "manager": manager,
            "signature_authority_limit": sig_limit,
            "address": rng.choice(_EMPLOYEE_ADDRESSES),
        })

    # --- Vendors ---
    n_vendors = rng.randint(30, 45)
    vendor_names_local = random_names(rand_seed + 100, n_vendors)
    # Append LLC/Inc suffixes
    suffixes = [" LLC", " Inc.", " Corp.", " Services", " Solutions", " Group", " Partners"]
    vendors: list[dict] = []
    for i, name in enumerate(vendor_names_local):
        vname = name.replace(" ", "") + rng.choice(suffixes)
        date_added_year = rng.randint(2018, 2023)
        date_added_month = rng.randint(1, 12)
        tax_id = f"{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}"
        bank_acct = f"****{rng.randint(1000, 9999)}"
        contact = random_name(rand_seed + 200 + i)
        vendors.append({
            "vendor_id": f"VND-{2000 + i}",
            "name": vname,
            "address": rng.choice(_VENDOR_ADDRESSES),
            "tax_id": tax_id,
            "bank_account": bank_acct,
            "date_added": f"{date_added_year}-{date_added_month:02d}-01",
            "contact_person": contact,
            "status": "Active",
        })

    # --- Select fraud patterns ---
    n_frauds = rng.randint(2, 4)
    chosen_frauds = rng.sample(_FRAUD_PATTERNS, n_frauds)

    # Assign specific employees and vendors to each fraud pattern
    fraud_details: list[dict] = []

    # Pick employees and vendors for fraud (ensure distinct)
    fraud_employee_indices = rng.sample(range(n_employees), min(n_frauds + 2, n_employees))
    fraud_vendor_indices = rng.sample(range(n_vendors), min(n_frauds + 2, n_vendors))

    # Track clean vendors for false-positive checks
    all_fraud_vendor_ids: set[str] = set()

    for fi, fraud in enumerate(chosen_frauds):
        emp_idx = fraud_employee_indices[fi]
        vnd_idx = fraud_vendor_indices[fi]
        emp = employees[emp_idx]
        vnd = vendors[vnd_idx]

        detail: dict = {
            "pattern": fraud["key"],
            "label": fraud["label"],
            "description": fraud["description"],
        }

        if fraud["key"] == "shell_company":
            # Make vendor address match employee address
            shared_address = emp["address"]
            vnd["address"] = shared_address
            detail["vendor_id"] = vnd["vendor_id"]
            detail["vendor_name"] = vnd["name"]
            detail["employee_id"] = emp["id"]
            detail["employee_name"] = emp["name"]
            detail["shared_address"] = shared_address
            detail["finding"] = (
                f"Vendor {vnd['vendor_id']} ({vnd['name']}) has address '{shared_address}' "
                f"which matches employee {emp['id']} ({emp['name']})"
            )
            all_fraud_vendor_ids.add(vnd["vendor_id"])

        elif fraud["key"] == "structuring":
            # Employee splits transactions to stay under their limit
            limit = emp["signature_authority_limit"]
            total_intended = round(rng.uniform(limit * 1.5, limit * 3.0), 2)
            n_splits = int(total_intended // (limit * 0.9)) + 1
            split_amounts = []
            remaining = total_intended
            for s in range(n_splits - 1):
                amt = round(rng.uniform(limit * 0.7, limit * 0.95), 2)
                split_amounts.append(amt)
                remaining -= amt
            remaining = round(remaining, 2)
            if remaining < 100.0:
                # Adjust the penultimate split to keep total correct
                shortfall = 100.0 - remaining
                split_amounts[-1] = round(split_amounts[-1] - shortfall, 2)
                remaining = 100.0
            split_amounts.append(remaining)
            detail["employee_id"] = emp["id"]
            detail["employee_name"] = emp["name"]
            detail["vendor_id"] = vnd["vendor_id"]
            detail["vendor_name"] = vnd["name"]
            detail["approval_limit"] = limit
            detail["split_amounts"] = split_amounts
            detail["total_amount"] = round(sum(split_amounts), 2)
            detail["n_splits"] = len(split_amounts)
            detail["finding"] = (
                f"Employee {emp['id']} ({emp['name']}) approved {len(split_amounts)} "
                f"transactions to vendor {vnd['vendor_id']} ({vnd['name']}) totaling "
                f"{_fmt_money(sum(split_amounts))}, each just under the {_fmt_money(limit)} "
                f"signature authority limit"
            )
            all_fraud_vendor_ids.add(vnd["vendor_id"])

        elif fraud["key"] == "ghost_vendor":
            # Make vendor date_added old, no recent activity until sudden payments
            vnd["date_added"] = f"{rng.randint(2018, 2020)}-{rng.randint(1, 12):02d}-01"
            ghost_amount = round(rng.uniform(15000, 75000), 2)
            detail["vendor_id"] = vnd["vendor_id"]
            detail["vendor_name"] = vnd["name"]
            detail["date_added"] = vnd["date_added"]
            detail["ghost_amount"] = ghost_amount
            detail["finding"] = (
                f"Vendor {vnd['vendor_id']} ({vnd['name']}) added on {vnd['date_added']} "
                f"had no transactions until 2024 when {_fmt_money(ghost_amount)} in payments appeared"
            )
            all_fraud_vendor_ids.add(vnd["vendor_id"])

        elif fraud["key"] == "kickback":
            # All transactions to this vendor approved by same person
            kickback_approver = emp
            n_kickback_txns = rng.randint(8, 15)
            # Consume the same number of rng draws to keep downstream state consistent
            for _ in range(n_kickback_txns):
                rng.uniform(2000, 12000)
            detail["vendor_id"] = vnd["vendor_id"]
            detail["vendor_name"] = vnd["name"]
            detail["approver_id"] = kickback_approver["id"]
            detail["approver_name"] = kickback_approver["name"]
            detail["n_transactions"] = n_kickback_txns
            # total_amount and finding will be filled after transactions are generated
            detail["total_amount"] = 0.0
            detail["finding"] = ""
            all_fraud_vendor_ids.add(vnd["vendor_id"])

        elif fraud["key"] == "duplicate_invoice":
            dup_amount = round(rng.uniform(3000, 20000), 2)
            dup_day1 = rng.randint(1, 15)
            dup_day2_raw = dup_day1 + rng.randint(3, 20)
            dup_day2 = min(dup_day2_raw, 28)
            dup_month = rng.randint(1, 10)
            detail["vendor_id"] = vnd["vendor_id"]
            detail["vendor_name"] = vnd["name"]
            detail["duplicate_amount"] = dup_amount
            detail["date1"] = f"2024-{dup_month:02d}-{dup_day1:02d}"
            detail["date2"] = f"2024-{dup_month:02d}-{dup_day2:02d}"
            actual_gap = dup_day2 - dup_day1
            detail["finding"] = (
                f"Vendor {vnd['vendor_id']} ({vnd['name']}) has two invoices for exactly "
                f"{_fmt_money(dup_amount)} within {actual_gap} days "
                f"({detail['date1']} and {detail['date2']})"
            )
            all_fraud_vendor_ids.add(vnd["vendor_id"])

        elif fraud["key"] == "weekend_approvals":
            n_weekend = rng.randint(4, 8)
            # Consume the same number of rng draws to keep downstream state consistent
            for _ in range(n_weekend):
                rng.uniform(1000, 8000)
            detail["employee_id"] = emp["id"]
            detail["employee_name"] = emp["name"]
            detail["n_weekend_txns"] = n_weekend
            # total_amount and finding will be filled after transactions are generated
            detail["total_amount"] = 0.0
            detail["finding"] = ""

        fraud_details.append(detail)

    # --- Generate transactions ---
    transactions: list[dict] = []
    txn_id = 10000

    # Normal transactions
    n_normal_txns = rng.randint(180, 350)
    for _ in range(n_normal_txns):
        txn_id += 1
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        # Weekday only for normal transactions (Mon=0, Sun=6)
        # Simple heuristic: avoid specific date checks, just ensure they're normal
        vnd = rng.choice(vendors)
        emp = rng.choice(employees)
        amount = round(rng.uniform(100, emp["signature_authority_limit"] * 0.8), 2)
        cat = rng.choice(_EXPENSE_CATEGORIES)
        transactions.append({
            "txn_id": f"TXN-{txn_id}",
            "date": f"2024-{month:02d}-{day:02d}",
            "vendor_id": vnd["vendor_id"],
            "amount": amount,
            "category": cat,
            "approved_by": emp["id"],
            "department": emp["department"],
            "description": f"{cat} - {vnd['name']}",
        })

    # Fraud-specific transactions
    for fd in fraud_details:
        if fd["pattern"] == "structuring":
            for i, amt in enumerate(fd["split_amounts"]):
                txn_id += 1
                month = rng.randint(3, 8)
                day = rng.randint(1, 28)
                transactions.append({
                    "txn_id": f"TXN-{txn_id}",
                    "date": f"2024-{month:02d}-{day:02d}",
                    "vendor_id": fd["vendor_id"],
                    "amount": round(amt, 2),
                    "category": rng.choice(_EXPENSE_CATEGORIES),
                    "approved_by": fd["employee_id"],
                    "department": next(e["department"] for e in employees if e["id"] == fd["employee_id"]),
                    "description": f"Service payment - {fd['vendor_name']} (part {i+1})",
                })

        elif fd["pattern"] == "ghost_vendor":
            # Sudden payments in 2024
            for _ in range(rng.randint(2, 4)):
                txn_id += 1
                month = rng.randint(6, 11)
                day = rng.randint(1, 28)
                amt = round(fd["ghost_amount"] / rng.randint(2, 4), 2)
                emp = rng.choice(employees)
                transactions.append({
                    "txn_id": f"TXN-{txn_id}",
                    "date": f"2024-{month:02d}-{day:02d}",
                    "vendor_id": fd["vendor_id"],
                    "amount": amt,
                    "category": "Professional Services",
                    "approved_by": emp["id"],
                    "department": emp["department"],
                    "description": f"Consulting services - {fd['vendor_name']}",
                })

        elif fd["pattern"] == "kickback":
            for _ in range(fd["n_transactions"]):
                txn_id += 1
                month = rng.randint(1, 12)
                day = rng.randint(1, 28)
                amt = round(rng.uniform(2000, 12000), 2)
                transactions.append({
                    "txn_id": f"TXN-{txn_id}",
                    "date": f"2024-{month:02d}-{day:02d}",
                    "vendor_id": fd["vendor_id"],
                    "amount": amt,
                    "category": rng.choice(_EXPENSE_CATEGORIES),
                    "approved_by": fd["approver_id"],
                    "department": rng.choice(_DEPARTMENTS),
                    "description": f"Service delivery - {fd['vendor_name']}",
                })

        elif fd["pattern"] == "duplicate_invoice":
            for date_str in [fd["date1"], fd["date2"]]:
                txn_id += 1
                emp = rng.choice(employees)
                transactions.append({
                    "txn_id": f"TXN-{txn_id}",
                    "date": date_str,
                    "vendor_id": fd["vendor_id"],
                    "amount": fd["duplicate_amount"],
                    "category": "Professional Services",
                    "approved_by": emp["id"],
                    "department": emp["department"],
                    "description": f"Invoice payment - {fd['vendor_name']}",
                })

        elif fd["pattern"] == "weekend_approvals":
            # Saturdays in 2024: Jan 6, 13, 20, 27; Feb 3, 10, ...
            # Use fixed weekend dates
            weekend_dates = [
                "2024-01-06", "2024-02-10", "2024-03-16", "2024-04-20",
                "2024-05-11", "2024-06-15", "2024-07-20", "2024-08-17",
                "2024-09-14", "2024-10-19", "2024-11-16", "2024-12-21",
            ]
            chosen_dates = rng.sample(weekend_dates, min(fd["n_weekend_txns"], len(weekend_dates)))
            for wd in chosen_dates:
                txn_id += 1
                vnd = rng.choice(vendors)
                amt = round(rng.uniform(1000, 8000), 2)
                transactions.append({
                    "txn_id": f"TXN-{txn_id}",
                    "date": wd,
                    "vendor_id": vnd["vendor_id"],
                    "amount": amt,
                    "category": rng.choice(_EXPENSE_CATEGORIES),
                    "approved_by": fd["employee_id"],
                    "department": next(e["department"] for e in employees if e["id"] == fd["employee_id"]),
                    "description": f"Urgent payment - {vnd['name']}",
                })

        elif fd["pattern"] == "shell_company":
            # Normal-looking transactions to the shell vendor
            for _ in range(rng.randint(4, 8)):
                txn_id += 1
                month = rng.randint(1, 12)
                day = rng.randint(1, 28)
                amt = round(rng.uniform(3000, 15000), 2)
                transactions.append({
                    "txn_id": f"TXN-{txn_id}",
                    "date": f"2024-{month:02d}-{day:02d}",
                    "vendor_id": fd["vendor_id"],
                    "amount": amt,
                    "category": "Consulting",
                    "approved_by": fd["employee_id"],
                    "department": next(e["department"] for e in employees if e["id"] == fd["employee_id"]),
                    "description": f"Consulting engagement - {fd['vendor_name']}",
                })

    # Backfill correct totals for fraud patterns whose pre-computed totals
    # don't match the actual generated transactions.
    for fd in fraud_details:
        if fd["pattern"] == "kickback":
            actual_total = round(sum(
                t["amount"] for t in transactions
                if t["vendor_id"] == fd["vendor_id"] and t["approved_by"] == fd["approver_id"]
            ), 2)
            fd["total_amount"] = actual_total
            fd["finding"] = (
                f"All {fd['n_transactions']} transactions to vendor {fd['vendor_id']} ({fd['vendor_name']}) "
                f"totaling {_fmt_money(actual_total)} were approved exclusively by "
                f"{fd['approver_id']} ({fd['approver_name']})"
            )
        elif fd["pattern"] == "weekend_approvals":
            actual_total = round(sum(
                t["amount"] for t in transactions
                if t["approved_by"] == fd["employee_id"] and t["date"] in {
                    "2024-01-06", "2024-02-10", "2024-03-16", "2024-04-20",
                    "2024-05-11", "2024-06-15", "2024-07-20", "2024-08-17",
                    "2024-09-14", "2024-10-19", "2024-11-16", "2024-12-21",
                }
            ), 2)
            fd["total_amount"] = actual_total
            fd["finding"] = (
                f"Employee {fd['employee_id']} ({fd['employee_name']}) approved {fd['n_weekend_txns']} transactions "
                f"totaling {_fmt_money(actual_total)} on weekends/holidays"
            )
        elif fd["pattern"] == "ghost_vendor":
            actual_total = round(sum(
                t["amount"] for t in transactions
                if t["vendor_id"] == fd["vendor_id"]
            ), 2)
            fd["ghost_amount"] = actual_total
            fd["total_amount"] = actual_total
            fd["finding"] = (
                f"Vendor {fd['vendor_id']} ({fd['vendor_name']}) added on {fd['date_added']} "
                f"had no transactions until 2024 when {_fmt_money(actual_total)} in payments appeared"
            )

    # Shuffle transactions and sort by date
    rng.shuffle(transactions)
    transactions.sort(key=lambda t: t["date"])

    # --- Build CSV files ---

    # transactions.csv
    txn_header = "txn_id,date,vendor_id,amount,category,approved_by,department,description"
    txn_rows = [txn_header]
    for t in transactions:
        txn_rows.append(
            f"{t['txn_id']},{t['date']},{t['vendor_id']},{t['amount']:.2f},"
            f"{t['category']},{t['approved_by']},{t['department']},\"{t['description']}\""
        )
    transactions_content = "\n".join(txn_rows) + "\n"

    # vendor_master.csv
    vnd_header = "vendor_id,name,address,tax_id,bank_account,date_added,contact_person,status"
    vnd_rows = [vnd_header]
    for v in vendors:
        vnd_rows.append(
            f"{v['vendor_id']},{v['name']},\"{v['address']}\",{v['tax_id']},"
            f"{v['bank_account']},{v['date_added']},{v['contact_person']},{v['status']}"
        )
    vendor_content = "\n".join(vnd_rows) + "\n"

    # employee_directory.csv
    emp_header = "id,name,department,title,hire_date,manager,signature_authority_limit,address"
    emp_rows = [emp_header]
    for e in employees:
        emp_rows.append(
            f"{e['id']},{e['name']},{e['department']},{e['title']},"
            f"{e['hire_date']},{e['manager']},{e['signature_authority_limit']},\"{e['address']}\""
        )
    employee_content = "\n".join(emp_rows) + "\n"

    # policy.txt
    policy_lines = [
        f"{company} — Financial Controls and Approval Policy",
        f"Effective Date: January 1, 2024",
        "",
        "=" * 60,
        "SECTION 1: APPROVAL AUTHORITY",
        "=" * 60,
        "",
        "Transaction approval limits by role:",
        "  Analyst / Associate / Coordinator / Specialist:  up to $5,000",
        "  Manager / Senior Manager / Lead:                 up to $25,000",
        "  Director / VP:                                   up to $100,000",
        "  CFO / CEO:                                       unlimited",
        "",
        "Employees MUST NOT approve transactions exceeding their signature",
        "authority limit. Transactions exceeding the limit require approval",
        "from the next level of management.",
        "",
        "=" * 60,
        "SECTION 2: DUAL-SIGNATURE REQUIREMENTS",
        "=" * 60,
        "",
        "Transactions over $25,000 require dual signatures from two",
        "authorized signatories, at least one of whom must be Director-level",
        "or above.",
        "",
        "=" * 60,
        "SECTION 3: VENDOR ONBOARDING",
        "=" * 60,
        "",
        "3.1  New vendors must complete a W-9 verification process.",
        "3.2  Vendor addresses must be verified against employee address",
        "     records to prevent conflicts of interest.",
        "3.3  Vendors inactive for 12 or more consecutive months must be",
        "     re-verified before any new payments are processed.",
        "",
        "=" * 60,
        "SECTION 4: TRANSACTION MONITORING",
        "=" * 60,
        "",
        "4.1  Round-number payments over $1,000 (e.g., $5,000.00 exactly)",
        "     are flagged for additional review.",
        "4.2  Multiple transactions to the same vendor within 30 days with",
        "     identical amounts are flagged as potential duplicates.",
        "4.3  Transactions split across multiple entries that individually",
        "     fall below approval thresholds but collectively exceed them",
        "     are considered structuring and are prohibited.",
        "4.4  All approvals must occur during business hours (Mon-Fri,",
        "     8:00 AM - 6:00 PM). Weekend and holiday approvals require",
        "     documented emergency justification.",
        "4.5  No single approver should handle more than 60% of transactions",
        "     for any given vendor. Diversified approval is required.",
        "",
        "=" * 60,
        "SECTION 5: AUDIT AND REPORTING",
        "=" * 60,
        "",
        "5.1  Quarterly audits will review a random sample of transactions.",
        "5.2  Any pattern suggesting fraud (shell companies, kickbacks,",
        "     structuring, ghost vendors, duplicate payments) must be",
        "     reported to the Chief Compliance Officer immediately.",
        "",
    ]
    policy_content = "\n".join(policy_lines) + "\n"

    # Compute total suspicious amount across all fraud patterns
    total_suspicious = 0.0
    for fd in fraud_details:
        if fd["pattern"] == "structuring":
            total_suspicious += fd["total_amount"]
        elif fd["pattern"] == "ghost_vendor":
            total_suspicious += fd["ghost_amount"]
        elif fd["pattern"] == "kickback":
            total_suspicious += fd["total_amount"]
        elif fd["pattern"] == "duplicate_invoice":
            total_suspicious += fd["duplicate_amount"]  # One of the two is the duplicate
        elif fd["pattern"] == "weekend_approvals":
            total_suspicious += fd["total_amount"]
        elif fd["pattern"] == "shell_company":
            # Sum of transactions to shell vendor
            shell_txns = [t for t in transactions if t["vendor_id"] == fd["vendor_id"]]
            shell_total = sum(t["amount"] for t in shell_txns)
            fd["shell_total"] = round(shell_total, 2)
            total_suspicious += shell_total
    total_suspicious = round(total_suspicious, 2)

    # Pick clean vendors for false-positive checks
    clean_vendor_ids = [v["vendor_id"] for v in vendors if v["vendor_id"] not in all_fraud_vendor_ids]
    n_fp_checks = min(3, len(clean_vendor_ids))
    fp_vendor_ids = rng.sample(clean_vendor_ids, n_fp_checks) if clean_vendor_ids else []

    # --- Problem statement ---
    problem_statement = f"""# Financial Fraud Detection Audit

You are {auditor}, a forensic accountant at {company}. You have been assigned
to analyze transaction records, vendor master data, and employee information to
identify potential fraud patterns.

## Source Files
- /testbed/data/transactions.csv — Transaction records: date, vendor_id, amount, category, approved_by, department
- /testbed/data/vendor_master.csv — Vendor directory: vendor_id, name, address, tax_id, bank_account, date_added, contact, status
- /testbed/data/employee_directory.csv — Employee records: id, name, department, title, hire_date, manager, signature_authority_limit
- /testbed/data/policy.txt — Financial controls: approval limits, dual-signature rules, vendor onboarding, transaction monitoring flags

## Requirements
1. Cross-reference vendor addresses with employee addresses to identify conflicts of interest
2. Analyze transaction patterns for structuring (splitting to avoid approval limits)
3. Check for ghost vendors (inactive then suddenly receiving payments)
4. Look for kickback patterns (single approver monopolizing a vendor)
5. Identify duplicate invoices (same vendor + same amount within 30 days)
6. Check for weekend/holiday approvals without justification
7. Compute the total dollar amount of suspicious transactions
8. Clearly distinguish confirmed fraud indicators from legitimate activity

Write a detailed forensic audit report to /testbed/audit_report.txt with your
findings, evidence, and recommendations."""

    # --- Rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/audit_report.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
    ]

    # Per-fraud-pattern detection checks
    for i, fd in enumerate(fraud_details):
        rubric_items.append(
            BinaryRubricCategory(
                name=f"found_fraud_pattern_{i+1}",
                question=(
                    f"Does the audit report identify the following fraud pattern: "
                    f"{fd['label']}? Specifically: {fd['finding']}"
                ),
                points=3,
            )
        )

    # Per-fraud-pattern: correct vendor/employee identification
    for i, fd in enumerate(fraud_details):
        if "vendor_id" in fd:
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"correct_vendor_id_fraud_{i+1}",
                    question=(
                        f"Does the audit report correctly identify vendor {fd['vendor_id']} "
                        f"({fd.get('vendor_name', 'N/A')}) as involved in the "
                        f"'{fd['pattern'].replace('_', ' ')}' fraud pattern?"
                    ),
                    points=2,
                )
            )
        if "employee_id" in fd:
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"correct_employee_fraud_{i+1}",
                    question=(
                        f"Does the audit report correctly identify employee {fd['employee_id']} "
                        f"({fd.get('employee_name', 'N/A')}) as involved in the "
                        f"'{fd['pattern'].replace('_', ' ')}' fraud pattern?"
                    ),
                    points=2,
                )
            )

    # Specific amount checks per pattern
    for i, fd in enumerate(fraud_details):
        if fd["pattern"] == "structuring":
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"correct_structuring_total_{i+1}",
                    question=(
                        f"Does the audit report identify that the {fd['n_splits']} structured "
                        f"transactions total approximately {_fmt_money(fd['total_amount'])} "
                        f"(within 10%), each staying under the {_fmt_money(fd['approval_limit'])} "
                        f"approval limit?"
                    ),
                    points=2,
                )
            )
        elif fd["pattern"] == "duplicate_invoice":
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"correct_duplicate_amount_{i+1}",
                    question=(
                        f"Does the audit report identify the duplicate invoice amount as "
                        f"{_fmt_money(fd['duplicate_amount'])} appearing on {fd['date1']} "
                        f"and {fd['date2']}?"
                    ),
                    points=2,
                )
            )
        elif fd["pattern"] == "kickback":
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"correct_kickback_count_{i+1}",
                    question=(
                        f"Does the audit report note that {fd['approver_name']} approved "
                        f"all {fd['n_transactions']} transactions (or approximately that many) "
                        f"to vendor {fd['vendor_id']}?"
                    ),
                    points=2,
                )
            )

    # --- General / always-present rubric items ---

    # Transaction and vendor counts
    total_txn_amount = round(sum(t["amount"] for t in transactions), 2)
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_total_transaction_count",
            question=(
                f"Does the audit report correctly state or demonstrate that there are "
                f"{len(transactions)} transactions in the dataset (within +/- 5)?"
            ),
            points=1,
        )
    )
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_vendor_count",
            question=(
                f"Does the audit report correctly state or demonstrate that there are "
                f"{n_vendors} vendors in the vendor master data (within +/- 2)?"
            ),
            points=1,
        )
    )
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_total_transaction_amount",
            question=(
                f"Does the audit report state the total transaction volume as approximately "
                f"{_fmt_money(total_txn_amount)} (within 10%)?"
            ),
            points=1,
        )
    )

    # Total suspicious amount
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_total_suspicious_amount",
            question=(
                f"Does the audit report estimate the total suspicious transaction amount "
                f"as approximately {_fmt_money(total_suspicious)} (within 20%, i.e., between "
                f"{_fmt_money(total_suspicious * 0.8)} and {_fmt_money(total_suspicious * 1.2)})?"
            ),
            points=2,
        )
    )

    # Correct fraud pattern count
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_fraud_pattern_count",
            question=(
                f"Does the audit report identify exactly {n_frauds} distinct fraud patterns "
                f"(not more, not fewer)?"
            ),
            points=2,
        )
    )

    # False positive checks
    for vnd_id in fp_vendor_ids:
        vnd_name = next(v["name"] for v in vendors if v["vendor_id"] == vnd_id)
        rubric_items.append(
            BinaryRubricCategory(
                name=f"no_false_flag_{vnd_id}",
                question=(
                    f"Does the audit report correctly avoid flagging vendor {vnd_id} "
                    f"({vnd_name}) as fraudulent? This vendor has no fraud indicators."
                ),
                points=2,
            )
        )

    rubric_items.append(
        BinaryRubricCategory(
            name="cross_references_documents",
            question=(
                "Does the audit report demonstrate cross-referencing between at least two "
                "different source files (e.g., matching vendor addresses to employee addresses, "
                "or checking transaction amounts against employee approval limits)?"
            ),
            points=1,
        )
    )

    rubric_items.append(
        BinaryRubricCategory(
            name="cites_policy_sections",
            question=(
                "Does the audit report cite specific policy section numbers (e.g., "
                "Section 3.2, Section 4.3) when describing which rules were violated?"
            ),
            points=1,
        )
    )

    rubric_items.append(
        BinaryRubricCategory(
            name="has_executive_summary",
            question=(
                "Does the audit report include an executive summary or overview section "
                "that states the total number of fraud patterns found, the total dollar "
                "amount at risk, and the overall assessment?"
            ),
            points=1,
        )
    )

    rubric_items.append(
        BinaryRubricCategory(
            name="provides_recommendations",
            question=(
                "Does the audit report provide specific remediation recommendations for "
                "each identified fraud pattern (e.g., terminate vendor relationship, "
                "reassign approval authority, recover duplicate payment)?"
            ),
            points=1,
        )
    )

    rubric_items.append(
        BinaryRubricCategory(
            name="no_fabricated_data",
            question=(
                "Does the audit report avoid fabricating or hallucinating vendor IDs, "
                "employee IDs, transaction amounts, or dates that do not appear in the "
                "source documents?"
            ),
            points=2,
        )
    )

    rubric_items.append(
        RubricCategory(
            name="report_quality",
            description="Is the forensic audit report well-organized with clear evidence, professional language, and actionable recommendations?",
            failure="Disorganized output or raw data without analysis.",
            minor_failure="Some structure but missing key evidence citations or unclear reasoning.",
            minor_success="Reasonably organized with evidence for most findings.",
            success="Professional forensic report with executive summary, detailed findings with evidence, risk assessment, and remediation recommendations.",
            points=2,
        )
    )

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=tuple(rubric_items),
        submission_instructions="Write your completed audit report to /testbed/audit_report.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/transactions.csv": transactions_content,
            "/testbed/data/vendor_master.csv": vendor_content,
            "/testbed/data/employee_directory.csv": employee_content,
            "/testbed/data/policy.txt": policy_content,
        },
        problem_type="financial_fraud_detection",
    )


# =============================================================================
# DOMAIN: MEDICAL CHART REVIEW
# =============================================================================

_CONDITIONS = [
    {
        "key": "diabetes_t2",
        "label": "Type 2 Diabetes Mellitus",
        "icd10": "E11.9",
        "typical_meds": [
            ("Metformin", "500mg", "twice daily"),
            ("Glipizide", "5mg", "once daily"),
        ],
        "monitoring": ["HbA1c every 3 months", "Fasting glucose", "Renal function panel"],
        "target_hba1c": 7.0,
    },
    {
        "key": "hypertension",
        "label": "Essential Hypertension",
        "icd10": "I10",
        "typical_meds": [
            ("Lisinopril", "10mg", "once daily"),
            ("Amlodipine", "5mg", "once daily"),
        ],
        "monitoring": ["Blood pressure check", "BMP (electrolytes, creatinine)"],
        "target_bp": "< 140/90",
    },
    {
        "key": "afib",
        "label": "Atrial Fibrillation",
        "icd10": "I48.91",
        "typical_meds": [
            ("Warfarin", "5mg", "once daily"),
            ("Metoprolol", "25mg", "twice daily"),
        ],
        "monitoring": ["INR every 2-4 weeks", "ECG", "CBC"],
        "target_inr": (2.0, 3.0),
    },
    {
        "key": "copd",
        "label": "Chronic Obstructive Pulmonary Disease",
        "icd10": "J44.1",
        "typical_meds": [
            ("Tiotropium", "18mcg", "once daily inhaled"),
            ("Albuterol", "90mcg", "as needed inhaled"),
        ],
        "monitoring": ["Pulmonary function test annually", "SpO2", "Chest X-ray"],
        "target_fev1": "> 80% predicted",
    },
    {
        "key": "ckd",
        "label": "Chronic Kidney Disease, Stage 3",
        "icd10": "N18.3",
        "typical_meds": [
            ("Losartan", "50mg", "once daily"),
        ],
        "monitoring": ["eGFR every 3 months", "Urine albumin-to-creatinine ratio", "BMP"],
        "target_egfr": "> 30 mL/min",
    },
    {
        "key": "hypothyroid",
        "label": "Hypothyroidism",
        "icd10": "E03.9",
        "typical_meds": [
            ("Levothyroxine", "75mcg", "once daily"),
        ],
        "monitoring": ["TSH every 6-8 weeks after dose change, then annually", "Free T4"],
        "target_tsh": (0.5, 4.5),
    },
    {
        "key": "depression",
        "label": "Major Depressive Disorder",
        "icd10": "F33.1",
        "typical_meds": [
            ("Sertraline", "50mg", "once daily"),
        ],
        "monitoring": ["PHQ-9 screening", "Follow-up 4-6 weeks after dose change"],
        "target_phq9": "< 5",
    },
    {
        "key": "osteoarthritis",
        "label": "Osteoarthritis",
        "icd10": "M19.90",
        "typical_meds": [
            ("Naproxen", "500mg", "twice daily"),
            ("Acetaminophen", "650mg", "as needed"),
        ],
        "monitoring": ["Pain assessment", "Renal function if on NSAIDs long-term"],
        "target_pain": "< 4/10",
    },
]

_DRUG_INTERACTIONS = [
    {
        "drug_a": "Warfarin",
        "drug_b": "Naproxen",
        "severity": "HIGH",
        "effect": "NSAIDs increase bleeding risk with warfarin by inhibiting platelet function and potentially increasing INR. Combined use significantly raises risk of GI hemorrhage.",
    },
    {
        "drug_a": "Warfarin",
        "drug_b": "Sertraline",
        "severity": "MODERATE",
        "effect": "SSRIs may increase bleeding risk by inhibiting platelet serotonin uptake. Combined with warfarin, monitor INR more frequently.",
    },
    {
        "drug_a": "Lisinopril",
        "drug_b": "Losartan",
        "severity": "HIGH",
        "effect": "Dual RAAS blockade (ACE inhibitor + ARB) increases risk of hyperkalemia, hypotension, and renal impairment. Generally contraindicated.",
    },
    {
        "drug_a": "Metformin",
        "drug_b": "Losartan",
        "severity": "MODERATE",
        "effect": "Monitor renal function closely. Metformin dose may need adjustment if eGFR declines due to RAAS inhibition.",
    },
    {
        "drug_a": "Glipizide",
        "drug_b": "Metoprolol",
        "severity": "MODERATE",
        "effect": "Beta-blockers may mask hypoglycemia symptoms (tachycardia, tremor). Monitor blood glucose more frequently.",
    },
    {
        "drug_a": "Amlodipine",
        "drug_b": "Metoprolol",
        "severity": "LOW",
        "effect": "Additive hypotension and bradycardia. Monitor blood pressure and heart rate. Generally manageable.",
    },
    {
        "drug_a": "Naproxen",
        "drug_b": "Lisinopril",
        "severity": "MODERATE",
        "effect": "NSAIDs may reduce antihypertensive effect of ACE inhibitors and worsen renal function. Monitor blood pressure and creatinine.",
    },
    {
        "drug_a": "Sertraline",
        "drug_b": "Metoprolol",
        "severity": "LOW",
        "effect": "Sertraline may increase metoprolol levels via CYP2D6 inhibition. Monitor for bradycardia.",
    },
]

_PLANTED_ERRORS = [
    {
        "key": "drug_interaction_high",
        "label": "Unaddressed high-severity drug interaction",
        "description": "A high-severity drug interaction exists in the patient's medication list with no documented plan to address it.",
    },
    {
        "key": "lab_no_followup",
        "label": "Abnormal lab value without documented follow-up",
        "description": "A lab result is significantly out of range but no follow-up action or repeat test is documented.",
    },
    {
        "key": "dose_renal",
        "label": "Medication dosage inappropriate for kidney function",
        "description": "A medication requiring renal dose adjustment is prescribed at the standard dose despite impaired kidney function.",
    },
    {
        "key": "missing_monitoring",
        "label": "Missing required monitoring test",
        "description": "A required monitoring test for a prescribed medication has not been performed within the recommended timeframe.",
    },
    {
        "key": "code_mismatch",
        "label": "Diagnosis code does not match clinical findings",
        "description": "An ICD-10 code in the chart does not correspond to the documented clinical findings or lab results.",
    },
]


# =============================================================================
# 3. MEDICAL CHART REVIEW
# =============================================================================


def make_medical_chart_review(rand_seed: int = 42) -> RubricDatapoint:
    """Given patient medical records, lab results, a drug interaction database,
    and clinical guidelines, identify documentation errors, medication
    interactions, missing follow-ups, and coding discrepancies.

    Seed varies: patient age, conditions (2-3 selected), which planted errors
    exist, medication list, lab values, and which safe medication pairs serve
    as false-positive checks.
    """
    rng = _random.Random(rand_seed)

    patient_name = random_name(rand_seed + 1)
    patient_age = rng.randint(52, 78)
    patient_sex = rng.choice(["Male", "Female"])
    reviewer_name = random_name(rand_seed + 2)

    # Select 2-3 conditions
    n_conditions = rng.randint(2, 3)
    chosen_conditions = rng.sample(_CONDITIONS, n_conditions)
    condition_keys = {c["key"] for c in chosen_conditions}

    # Build medication list from chosen conditions
    all_meds: list[dict] = []
    for cond in chosen_conditions:
        for med_name, dose, freq in cond["typical_meds"]:
            all_meds.append({
                "name": med_name,
                "dose": dose,
                "frequency": freq,
                "for_condition": cond["label"],
            })

    # Add 2-4 extra common medications as filler
    filler_meds = [
        ("Omeprazole", "20mg", "once daily", "GERD prophylaxis"),
        ("Vitamin D3", "1000 IU", "once daily", "Supplement"),
        ("Atorvastatin", "40mg", "once daily at bedtime", "Hyperlipidemia"),
        ("Aspirin", "81mg", "once daily", "Cardiovascular prophylaxis"),
        ("Calcium Carbonate", "500mg", "twice daily", "Supplement"),
        ("Gabapentin", "300mg", "three times daily", "Neuropathic pain"),
    ]
    n_filler = rng.randint(2, 4)
    for med_name, dose, freq, reason in rng.sample(filler_meds, n_filler):
        all_meds.append({
            "name": med_name,
            "dose": dose,
            "frequency": freq,
            "for_condition": reason,
        })

    # Determine which errors to plant based on available conditions
    possible_errors: list[dict] = []

    # High drug interaction: need warfarin + NSAID, or dual RAAS
    patient_med_names = {m["name"] for m in all_meds}
    has_warfarin = "Warfarin" in patient_med_names
    has_nsaid = "Naproxen" in patient_med_names
    has_lisinopril = "Lisinopril" in patient_med_names
    has_losartan = "Losartan" in patient_med_names

    planted_interaction = None
    if has_warfarin and has_nsaid:
        planted_interaction = next(
            d for d in _DRUG_INTERACTIONS
            if d["drug_a"] == "Warfarin" and d["drug_b"] == "Naproxen"
        )
        possible_errors.append({
            "error_key": "drug_interaction_high",
            "detail": (
                f"High-severity interaction between {planted_interaction['drug_a']} and "
                f"{planted_interaction['drug_b']}: {planted_interaction['effect']}"
            ),
            "severity": "HIGH",
            "drug_a": planted_interaction["drug_a"],
            "drug_b": planted_interaction["drug_b"],
        })
    elif has_lisinopril and has_losartan:
        planted_interaction = next(
            d for d in _DRUG_INTERACTIONS
            if d["drug_a"] == "Lisinopril" and d["drug_b"] == "Losartan"
        )
        possible_errors.append({
            "error_key": "drug_interaction_high",
            "detail": (
                f"High-severity interaction between {planted_interaction['drug_a']} and "
                f"{planted_interaction['drug_b']}: {planted_interaction['effect']}"
            ),
            "severity": "HIGH",
            "drug_a": planted_interaction["drug_a"],
            "drug_b": planted_interaction["drug_b"],
        })

    # Lab no followup: plant an abnormal lab value
    lab_abnormality = None
    if "diabetes_t2" in condition_keys:
        abnormal_hba1c = round(rng.uniform(8.5, 11.0), 1)
        lab_abnormality = {
            "test": "HbA1c",
            "value": abnormal_hba1c,
            "unit": "%",
            "ref_range": "4.0-5.6 (normal), < 7.0 (target for diabetes)",
            "flag": "HIGH",
            "clinical_note": f"HbA1c of {abnormal_hba1c}% indicates poorly controlled diabetes. No medication adjustment or follow-up plan documented.",
        }
        possible_errors.append({
            "error_key": "lab_no_followup",
            "detail": f"HbA1c is {abnormal_hba1c}% (target < 7.0% for diabetics) with no documented follow-up or medication adjustment",
            "test_name": "HbA1c",
            "value": abnormal_hba1c,
            "target": "< 7.0%",
        })
    elif "hypothyroid" in condition_keys:
        abnormal_tsh = round(rng.uniform(8.0, 15.0), 2)
        lab_abnormality = {
            "test": "TSH",
            "value": abnormal_tsh,
            "unit": "mIU/L",
            "ref_range": "0.5-4.5",
            "flag": "HIGH",
            "clinical_note": f"TSH of {abnormal_tsh} mIU/L indicates inadequate thyroid replacement. No dose adjustment documented.",
        }
        possible_errors.append({
            "error_key": "lab_no_followup",
            "detail": f"TSH is {abnormal_tsh} mIU/L (reference 0.5-4.5) with no documented follow-up or dose adjustment",
            "test_name": "TSH",
            "value": abnormal_tsh,
            "target": "0.5-4.5 mIU/L",
        })

    # Dose renal: if CKD is present and Metformin is prescribed
    if "ckd" in condition_keys and "Metformin" in patient_med_names:
        low_egfr = round(rng.uniform(25, 38), 1)
        possible_errors.append({
            "error_key": "dose_renal",
            "detail": (
                f"Metformin 500mg twice daily prescribed but eGFR is {low_egfr} mL/min "
                f"(Stage 3 CKD). Metformin should be dose-reduced or discontinued when "
                f"eGFR < 30 mL/min, and maximum dose limited to 1000mg/day when eGFR 30-45."
            ),
            "medication": "Metformin",
            "egfr": low_egfr,
        })
    else:
        low_egfr = None

    # Missing monitoring: if afib with warfarin, missing INR check
    if "afib" in condition_keys and has_warfarin:
        possible_errors.append({
            "error_key": "missing_monitoring",
            "detail": (
                "Patient is on Warfarin for atrial fibrillation but no INR test result "
                "appears in the last 4 weeks of lab data. INR should be monitored every "
                "2-4 weeks for patients on Warfarin."
            ),
            "medication": "Warfarin",
            "test": "INR",
        })

    # Code mismatch: plant a wrong ICD-10 code
    # Pick a condition and swap its code
    mismatch_condition = rng.choice(chosen_conditions)
    wrong_code_pool = [c for c in _CONDITIONS if c["key"] != mismatch_condition["key"]]
    wrong_code_source = rng.choice(wrong_code_pool)
    possible_errors.append({
        "error_key": "code_mismatch",
        "detail": (
            f"Chart lists ICD-10 code {wrong_code_source['icd10']} "
            f"({wrong_code_source['label']}) but clinical findings and medications "
            f"are consistent with {mismatch_condition['label']} "
            f"(correct code: {mismatch_condition['icd10']})"
        ),
        "wrong_code": wrong_code_source["icd10"],
        "wrong_label": wrong_code_source["label"],
        "correct_code": mismatch_condition["icd10"],
        "correct_label": mismatch_condition["label"],
    })

    # Select which errors to actually plant (keep all possible ones; minimum 3)
    if len(possible_errors) > 5:
        planted_errors = rng.sample(possible_errors, rng.randint(3, min(5, len(possible_errors))))
    else:
        planted_errors = possible_errors

    planted_error_keys = {e["error_key"] for e in planted_errors}

    # --- Build patient_chart.txt ---
    chart_lines = [
        "PATIENT MEDICAL CHART",
        "",
        "=" * 60,
        "DEMOGRAPHICS",
        "=" * 60,
        "",
        f"Name: {patient_name}",
        f"Age: {patient_age}",
        f"Sex: {patient_sex}",
        f"DOB: {2024 - patient_age}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        f"MRN: MRN-{rng.randint(100000, 999999)}",
        "",
        "=" * 60,
        "CHIEF COMPLAINT",
        "=" * 60,
        "",
        f"Routine follow-up for chronic condition management.",
        "",
        "=" * 60,
        "HISTORY OF PRESENT ILLNESS",
        "=" * 60,
        "",
        f"{patient_name} is a {patient_age}-year-old {patient_sex.lower()} presenting for",
        f"routine follow-up. Patient reports general compliance with medications",
        f"but notes occasional missed doses. Denies chest pain, shortness of",
        f"breath at rest, or acute changes in condition.",
        "",
        "=" * 60,
        "PAST MEDICAL HISTORY",
        "=" * 60,
        "",
    ]

    # List conditions with ICD-10 codes (with planted mismatch)
    for cond in chosen_conditions:
        if "code_mismatch" in planted_error_keys and cond["key"] == mismatch_condition["key"]:
            # Use the wrong code
            wrong_error = next(e for e in planted_errors if e["error_key"] == "code_mismatch")
            chart_lines.append(f"- {cond['label']} (ICD-10: {wrong_error['wrong_code']})")
        else:
            chart_lines.append(f"- {cond['label']} (ICD-10: {cond['icd10']})")

    chart_lines.extend([
        "",
        "=" * 60,
        "ALLERGIES",
        "=" * 60,
        "",
        f"- Penicillin (rash)",
        f"- Sulfa drugs (hives)",
        "",
        "=" * 60,
        "CURRENT MEDICATIONS",
        "=" * 60,
        "",
    ])

    for med in all_meds:
        chart_lines.append(f"- {med['name']} {med['dose']} {med['frequency']}  [{med['for_condition']}]")

    chart_lines.extend([
        "",
        "=" * 60,
        "PHYSICAL EXAMINATION",
        "=" * 60,
        "",
        f"Vitals:",
        f"  BP: {rng.randint(125, 155)}/{rng.randint(75, 95)} mmHg",
        f"  HR: {rng.randint(62, 95)} bpm",
        f"  Temp: {round(rng.uniform(97.8, 99.0), 1)}°F",
        f"  SpO2: {rng.randint(93, 99)}%",
        f"  Weight: {rng.randint(150, 220)} lbs",
        "",
        "General: Alert, oriented, no acute distress.",
        "HEENT: Normocephalic, atraumatic. Oropharynx clear.",
        "Cardiovascular: Regular rate and rhythm, no murmurs." if "afib" not in condition_keys
        else "Cardiovascular: Irregularly irregular rhythm, no murmurs.",
        "Lungs: " + ("Clear to auscultation bilaterally." if "copd" not in condition_keys
                      else "Mild expiratory wheezing bilaterally, decreased breath sounds at bases."),
        "Extremities: No edema, pulses 2+ bilaterally.",
        "",
        "=" * 60,
        "ASSESSMENT AND PLAN",
        "=" * 60,
        "",
        "Continue current medication regimen. Follow up in 3 months.",
        "Patient counseled on medication compliance and lifestyle modifications.",
        "",
    ])

    chart_content = "\n".join(chart_lines) + "\n"

    # --- Build lab_results.csv ---
    lab_header = "date,test_name,value,unit,reference_range,flag"
    lab_rows = [lab_header]

    visit_dates = ["2024-01-15", "2024-04-22", "2024-07-10"]

    for visit_date in visit_dates:
        # Basic metabolic panel
        creatinine = round(rng.uniform(0.8, 1.4), 2)
        if "ckd" in condition_keys and low_egfr is not None and "dose_renal" in planted_error_keys:
            creatinine = round(rng.uniform(1.8, 2.5), 2)  # elevated for CKD
        bun = round(rng.uniform(10, 25), 1)
        sodium = rng.randint(136, 145)
        potassium = round(rng.uniform(3.5, 5.2), 1)
        glucose = rng.randint(90, 180) if "diabetes_t2" in condition_keys else rng.randint(75, 110)

        lab_rows.append(f"{visit_date},Creatinine,{creatinine},mg/dL,0.7-1.3,{'HIGH' if creatinine > 1.3 else 'NORMAL'}")
        lab_rows.append(f"{visit_date},BUN,{bun},mg/dL,7-20,{'HIGH' if bun > 20 else 'NORMAL'}")
        lab_rows.append(f"{visit_date},Sodium,{sodium},mEq/L,136-145,NORMAL")
        lab_rows.append(f"{visit_date},Potassium,{potassium},mEq/L,3.5-5.0,{'HIGH' if potassium > 5.0 else 'NORMAL'}")
        lab_rows.append(f"{visit_date},Glucose,{glucose},mg/dL,70-100,{'HIGH' if glucose > 100 else 'NORMAL'}")

        # eGFR (computed from creatinine)
        if "ckd" in condition_keys and low_egfr is not None and "dose_renal" in planted_error_keys:
            egfr_val = low_egfr
        else:
            egfr_val = round(rng.uniform(60, 95), 1)
        lab_rows.append(f"{visit_date},eGFR,{egfr_val},mL/min/1.73m2,> 60,{'LOW' if egfr_val < 60 else 'NORMAL'}")

        # HbA1c for diabetics
        if "diabetes_t2" in condition_keys:
            if "lab_no_followup" in planted_error_keys and visit_date == visit_dates[-1]:
                hba1c_val = lab_abnormality["value"]
            else:
                hba1c_val = round(rng.uniform(6.5, 7.8), 1)
            lab_rows.append(f"{visit_date},HbA1c,{hba1c_val},%,4.0-5.6,{'HIGH' if hba1c_val > 5.6 else 'NORMAL'}")

        # TSH for hypothyroid
        if "hypothyroid" in condition_keys:
            if "lab_no_followup" in planted_error_keys and lab_abnormality and lab_abnormality["test"] == "TSH" and visit_date == visit_dates[-1]:
                tsh_val = lab_abnormality["value"]
            else:
                tsh_val = round(rng.uniform(1.0, 4.0), 2)
            lab_rows.append(f"{visit_date},TSH,{tsh_val},mIU/L,0.5-4.5,{'HIGH' if tsh_val > 4.5 else 'NORMAL'}")

        # INR for warfarin patients
        if has_warfarin:
            if "missing_monitoring" in planted_error_keys and visit_date == visit_dates[-1]:
                pass  # Skip INR on last visit (the planted error)
            else:
                inr_val = round(rng.uniform(1.8, 3.2), 1)
                lab_rows.append(f"{visit_date},INR,{inr_val},,2.0-3.0,{'HIGH' if inr_val > 3.0 else 'LOW' if inr_val < 2.0 else 'NORMAL'}")

        # CBC (common)
        wbc = round(rng.uniform(4.5, 11.0), 1)
        hgb = round(rng.uniform(12.0, 16.0), 1) if patient_sex == "Male" else round(rng.uniform(11.0, 15.0), 1)
        plt = rng.randint(150, 400)
        lab_rows.append(f"{visit_date},WBC,{wbc},x10^3/uL,4.5-11.0,NORMAL")
        lab_rows.append(f"{visit_date},Hemoglobin,{hgb},g/dL,{'13.5-17.5' if patient_sex == 'Male' else '12.0-16.0'},NORMAL")
        lab_rows.append(f"{visit_date},Platelets,{plt},x10^3/uL,150-400,NORMAL")

        # Lipid panel on first visit
        if visit_date == visit_dates[0]:
            total_chol = rng.randint(160, 240)
            ldl = rng.randint(80, 160)
            hdl = rng.randint(35, 70)
            trig = rng.randint(100, 250)
            lab_rows.append(f"{visit_date},Total Cholesterol,{total_chol},mg/dL,< 200,{'HIGH' if total_chol > 200 else 'NORMAL'}")
            lab_rows.append(f"{visit_date},LDL,{ldl},mg/dL,< 100,{'HIGH' if ldl > 100 else 'NORMAL'}")
            lab_rows.append(f"{visit_date},HDL,{hdl},mg/dL,> 40,{'LOW' if hdl < 40 else 'NORMAL'}")
            lab_rows.append(f"{visit_date},Triglycerides,{trig},mg/dL,< 150,{'HIGH' if trig > 150 else 'NORMAL'}")

    lab_content = "\n".join(lab_rows) + "\n"

    # --- Build medication_interactions.txt ---
    # Include all interactions relevant to patient's meds, plus some irrelevant ones
    interaction_lines = [
        "DRUG INTERACTION DATABASE — Subset for Current Patient Medications",
        "",
        "=" * 70,
        f"{'Drug A':<20} {'Drug B':<20} {'Severity':<12} Clinical Effect",
        "=" * 70,
        "",
    ]

    relevant_interactions: list[dict] = []
    for inter in _DRUG_INTERACTIONS:
        if inter["drug_a"] in patient_med_names or inter["drug_b"] in patient_med_names:
            relevant_interactions.append(inter)

    # Add all relevant interactions
    for inter in relevant_interactions:
        interaction_lines.append(
            f"{inter['drug_a']:<20} {inter['drug_b']:<20} {inter['severity']:<12} {inter['effect']}"
        )
        interaction_lines.append("")

    # Add a few that aren't relevant (noise)
    noise_interactions = [
        ("Digoxin", "Amiodarone", "HIGH", "Amiodarone increases digoxin levels by 70-100%. Reduce digoxin dose by 50% when initiating amiodarone."),
        ("Simvastatin", "Diltiazem", "HIGH", "Diltiazem increases statin levels, raising risk of myopathy/rhabdomyolysis. Limit simvastatin to 10mg/day."),
        ("Clopidogrel", "Omeprazole", "MODERATE", "Omeprazole may reduce clopidogrel efficacy via CYP2C19 inhibition. Consider pantoprazole instead."),
    ]
    for da, db, sev, eff in noise_interactions:
        interaction_lines.append(f"{da:<20} {db:<20} {sev:<12} {eff}")
        interaction_lines.append("")

    interaction_lines.append("=" * 70)
    interaction_lines.append("")
    interaction_lines.append("NOTE: This database contains interactions relevant to this patient's")
    interaction_lines.append("medication profile. Always verify with current pharmacological references.")
    interaction_lines.append("")
    interaction_content = "\n".join(interaction_lines) + "\n"

    # --- Build clinical_guidelines.txt ---
    guideline_lines = [
        "CLINICAL PRACTICE GUIDELINES — Summary for Chart Review",
        "",
    ]

    if "diabetes_t2" in condition_keys:
        guideline_lines.extend([
            "=" * 60,
            "DIABETES MANAGEMENT (ADA Standards of Care 2024)",
            "=" * 60,
            "",
            "- HbA1c target: < 7.0% for most adults (individualized)",
            "- HbA1c monitoring: every 3 months if not at target, every 6 months if stable",
            "- Fasting glucose target: 80-130 mg/dL",
            "- Metformin: first-line therapy; contraindicated if eGFR < 30 mL/min",
            "  Dose reduction recommended if eGFR 30-45 mL/min (max 1000mg/day)",
            "- Annual comprehensive metabolic panel, lipid panel, urine albumin",
            "- HbA1c > 9.0% warrants immediate medication adjustment consideration",
            "",
        ])

    if "hypertension" in condition_keys:
        guideline_lines.extend([
            "=" * 60,
            "HYPERTENSION MANAGEMENT (AHA/ACC 2024)",
            "=" * 60,
            "",
            "- BP target: < 140/90 mmHg (general); < 130/80 for high CV risk",
            "- ACE inhibitors (e.g., lisinopril) and ARBs (e.g., losartan) should",
            "  NOT be used together (dual RAAS blockade contraindicated)",
            "- Monitor electrolytes and creatinine within 2 weeks of starting/changing",
            "  ACE inhibitor or ARB",
            "- NSAIDs may reduce effectiveness of antihypertensives",
            "",
        ])

    if "afib" in condition_keys:
        guideline_lines.extend([
            "=" * 60,
            "ATRIAL FIBRILLATION — ANTICOAGULATION (AHA/ACC/HRS 2024)",
            "=" * 60,
            "",
            "- Warfarin: target INR 2.0-3.0",
            "- INR monitoring: every 2-4 weeks while on warfarin",
            "- If INR not checked within 4 weeks, flag as overdue monitoring",
            "- Avoid concomitant NSAIDs (increased bleeding risk)",
            "- SSRIs may increase bleeding risk; monitor INR more frequently if added",
            "- Rate control: resting HR target < 110 bpm",
            "",
        ])

    if "ckd" in condition_keys:
        guideline_lines.extend([
            "=" * 60,
            "CHRONIC KIDNEY DISEASE (KDIGO 2024)",
            "=" * 60,
            "",
            "- eGFR monitoring: every 3 months for Stage 3",
            "- Stage 3a: eGFR 45-59; Stage 3b: eGFR 30-44",
            "- Metformin: reduce dose if eGFR 30-45; discontinue if eGFR < 30",
            "- NSAIDs: avoid or use with extreme caution in CKD",
            "- ACE inhibitors/ARBs: monitor potassium and creatinine closely",
            "- Refer to nephrology if eGFR < 30 or rapidly declining",
            "",
        ])

    if "hypothyroid" in condition_keys:
        guideline_lines.extend([
            "=" * 60,
            "HYPOTHYROIDISM MANAGEMENT (ATA 2024)",
            "=" * 60,
            "",
            "- TSH target: 0.5-4.5 mIU/L (some guidelines prefer 0.5-2.5 for symptomatic patients)",
            "- Check TSH 6-8 weeks after dose change",
            "- Once stable, check TSH annually",
            "- TSH > 10 mIU/L: always treat; TSH 4.5-10: consider if symptomatic",
            "- Levothyroxine: take on empty stomach, 30-60 min before breakfast",
            "",
        ])

    if "copd" in condition_keys:
        guideline_lines.extend([
            "=" * 60,
            "COPD MANAGEMENT (GOLD 2024)",
            "=" * 60,
            "",
            "- Pulmonary function testing annually",
            "- Long-acting muscarinic antagonist (e.g., tiotropium) as maintenance",
            "- Short-acting beta-agonist (e.g., albuterol) for rescue",
            "- Annual influenza and pneumococcal vaccination",
            "- SpO2 target: > 90%",
            "",
        ])

    if "depression" in condition_keys:
        guideline_lines.extend([
            "=" * 60,
            "DEPRESSION MANAGEMENT (APA 2024)",
            "=" * 60,
            "",
            "- PHQ-9 screening at each visit",
            "- Follow-up 4-6 weeks after initiating or changing antidepressant",
            "- SSRIs (e.g., sertraline): first-line treatment",
            "- Monitor for serotonin syndrome if multiple serotonergic agents",
            "- SSRIs may increase bleeding risk, especially with anticoagulants",
            "",
        ])

    if "osteoarthritis" in condition_keys:
        guideline_lines.extend([
            "=" * 60,
            "OSTEOARTHRITIS MANAGEMENT (ACR 2024)",
            "=" * 60,
            "",
            "- NSAIDs (e.g., naproxen): use lowest effective dose for shortest duration",
            "- Monitor renal function and GI symptoms if on chronic NSAIDs",
            "- Avoid NSAIDs in patients with CKD (eGFR < 60) or on anticoagulants",
            "- Acetaminophen: safer alternative but limited efficacy",
            "- Consider topical NSAIDs to reduce systemic exposure",
            "",
        ])

    guideline_lines.extend([
        "=" * 60,
        "GENERAL DOCUMENTATION STANDARDS",
        "=" * 60,
        "",
        "- All ICD-10 codes must match documented clinical findings",
        "- Abnormal lab values require documented follow-up plan",
        "- Drug interactions must be acknowledged and addressed in chart notes",
        "- Medication dosing must account for renal and hepatic function",
        "",
    ])

    guidelines_content = "\n".join(guideline_lines) + "\n"

    # --- Problem statement ---
    conditions_str = ", ".join(c["label"] for c in chosen_conditions)
    problem_statement = f"""# Clinical Chart Review Audit

You are {reviewer_name}, a clinical auditor reviewing the medical chart for
patient {patient_name} ({patient_age}-year-old {patient_sex.lower()}) with
diagnoses including {conditions_str}.

## Source Files
- /testbed/data/patient_chart.txt — Patient demographics, history, physical exam, medications, and assessment
- /testbed/data/lab_results.csv — Lab values across 3 visits with reference ranges and flags
- /testbed/data/medication_interactions.txt — Drug interaction database for patient's medications
- /testbed/data/clinical_guidelines.txt — Relevant clinical practice guidelines

## Requirements
1. Review all medications for drug-drug interactions, checking severity levels
2. Cross-reference lab results with medications and clinical guidelines
3. Identify any abnormal lab values lacking documented follow-up
4. Check medication dosages against renal function (eGFR) and guidelines
5. Verify all ICD-10 diagnosis codes match clinical findings and medications
6. Check that required monitoring tests are documented (e.g., INR for warfarin)
7. Distinguish genuine issues from items that appear concerning but are clinically appropriate

Write a detailed chart review report to /testbed/chart_review.txt with your
findings, clinical rationale, and recommendations."""

    # --- Rubric ---
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does the file /testbed/chart_review.txt exist and contain substantive content (more than a few lines)?",
            points=1,
        ),
    ]

    # Per-planted-error detection
    for i, err in enumerate(planted_errors):
        rubric_items.append(
            BinaryRubricCategory(
                name=f"identifies_error_{i+1}_{err['error_key']}",
                question=f"Does the chart review identify the following issue: {err['detail']}?",
                points=3,
            )
        )

    # Specific checks per error type
    for err in planted_errors:
        if err["error_key"] == "drug_interaction_high":
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"correct_interaction_severity_{err['drug_a']}_{err['drug_b']}",
                    question=(
                        f"Does the chart review correctly identify the interaction between "
                        f"{err['drug_a']} and {err['drug_b']} as HIGH severity (or clinically "
                        f"significant / dangerous / contraindicated)?"
                    ),
                    points=2,
                )
            )
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"recommends_action_interaction_{err['drug_a']}_{err['drug_b']}",
                    question=(
                        f"Does the chart review recommend a specific action for the "
                        f"{err['drug_a']}/{err['drug_b']} interaction (e.g., discontinue one, "
                        f"substitute, or add monitoring)?"
                    ),
                    points=1,
                )
            )

        elif err["error_key"] == "lab_no_followup":
            rubric_items.append(
                BinaryRubricCategory(
                    name=f"correct_lab_value_{err['test_name']}",
                    question=(
                        f"Does the chart review cite the specific {err['test_name']} value "
                        f"of {err['value']} and note that it exceeds the target of {err['target']}?"
                    ),
                    points=2,
                )
            )

        elif err["error_key"] == "dose_renal":
            rubric_items.append(
                BinaryRubricCategory(
                    name="correct_egfr_concern",
                    question=(
                        f"Does the chart review identify that {err['medication']} dosing is "
                        f"inappropriate given the eGFR of {err['egfr']} mL/min, and explain "
                        f"the renal dosing guideline?"
                    ),
                    points=2,
                )
            )

        elif err["error_key"] == "missing_monitoring":
            rubric_items.append(
                BinaryRubricCategory(
                    name="correct_missing_test",
                    question=(
                        f"Does the chart review specifically identify that {err['test']} "
                        f"monitoring is missing from the most recent visit for patient on "
                        f"{err['medication']}?"
                    ),
                    points=2,
                )
            )

        elif err["error_key"] == "code_mismatch":
            rubric_items.append(
                BinaryRubricCategory(
                    name="correct_code_mismatch",
                    question=(
                        f"Does the chart review identify that ICD-10 code {err['wrong_code']} "
                        f"({err['wrong_label']}) is incorrect and should be "
                        f"{err['correct_code']} ({err['correct_label']})?"
                    ),
                    points=2,
                )
            )

    # False-positive checks: safe medication combinations that should NOT be flagged
    safe_pairs: list[tuple[str, str, str]] = []
    for inter in _DRUG_INTERACTIONS:
        if inter["severity"] == "LOW" and inter["drug_a"] in patient_med_names and inter["drug_b"] in patient_med_names:
            safe_pairs.append((inter["drug_a"], inter["drug_b"], inter["effect"]))

    for pair_a, pair_b, effect in safe_pairs[:2]:
        rubric_items.append(
            BinaryRubricCategory(
                name=f"no_false_interaction_{pair_a}_{pair_b}",
                question=(
                    f"Does the chart review correctly avoid flagging the {pair_a}/{pair_b} "
                    f"combination as a dangerous or high-severity interaction? This is a "
                    f"LOW-severity interaction that is generally manageable with monitoring."
                ),
                points=2,
            )
        )

    # Check that normal labs on early visits are not flagged as problems
    rubric_items.append(
        BinaryRubricCategory(
            name="no_false_flag_normal_cbc",
            question=(
                "Does the chart review correctly avoid flagging normal CBC values "
                "(WBC, Hemoglobin, Platelets within reference range) as abnormal?"
            ),
            points=1,
        )
    )

    rubric_items.append(
        BinaryRubricCategory(
            name="no_false_flag_electrolytes",
            question=(
                "Does the chart review correctly avoid flagging sodium and potassium "
                "values that fall within the normal reference range as abnormal?"
            ),
            points=1,
        )
    )

    # --- General / always-present rubric items ---

    # Medication count
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_medication_count",
            question=(
                f"Does the chart review correctly account for the patient's "
                f"{len(all_meds)} medications (within +/- 1) in its analysis?"
            ),
            points=1,
        )
    )

    # Condition count
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_condition_count",
            question=(
                f"Does the chart review identify all {n_conditions} of the patient's "
                f"active conditions ({conditions_str})?"
            ),
            points=1,
        )
    )

    # Allergy documentation
    rubric_items.append(
        BinaryRubricCategory(
            name="notes_allergies",
            question=(
                "Does the chart review note the patient's documented allergies "
                "(Penicillin and Sulfa drugs) and verify that no prescribed medications "
                "conflict with these allergies?"
            ),
            points=1,
        )
    )

    # Lab trend analysis
    rubric_items.append(
        BinaryRubricCategory(
            name="analyzes_lab_trends",
            question=(
                "Does the chart review analyze lab values across multiple visits "
                "(not just the most recent) to identify trends or changes over time?"
            ),
            points=2,
        )
    )

    # Overall error count
    rubric_items.append(
        BinaryRubricCategory(
            name="correct_error_count",
            question=(
                f"Does the chart review identify approximately {len(planted_errors)} "
                f"clinical issues or documentation errors (within +/- 1)?"
            ),
            points=2,
        )
    )

    # Cross-reference check
    rubric_items.append(
        BinaryRubricCategory(
            name="cross_references_documents",
            question=(
                "Does the chart review demonstrate cross-referencing between at least "
                "two different source files (e.g., checking lab values against medication "
                "guidelines, or comparing ICD-10 codes against the medication list)?"
            ),
            points=1,
        )
    )

    rubric_items.append(
        BinaryRubricCategory(
            name="provides_recommendations",
            question=(
                "Does the chart review provide specific clinical recommendations for "
                "each identified issue (e.g., discontinue medication, order lab test, "
                "adjust dosage, correct ICD-10 code)?"
            ),
            points=1,
        )
    )

    rubric_items.append(
        BinaryRubricCategory(
            name="no_fabricated_data",
            question=(
                "Does the chart review avoid fabricating or hallucinating lab values, "
                "medication names, dosages, or ICD-10 codes that do not appear in the "
                "source documents?"
            ),
            points=2,
        )
    )

    rubric_items.append(
        BinaryRubricCategory(
            name="reviews_all_three_visits",
            question=(
                "Does the chart review examine lab results from all three visit dates "
                f"({', '.join(visit_dates)}) rather than only the most recent?"
            ),
            points=1,
        )
    )

    rubric_items.append(
        BinaryRubricCategory(
            name="identifies_patient_demographics",
            question=(
                f"Does the chart review correctly state key patient demographics: "
                f"{patient_name}, {patient_age} years old, {patient_sex.lower()}?"
            ),
            points=1,
        )
    )

    rubric_items.append(
        BinaryRubricCategory(
            name="prioritizes_findings",
            question=(
                "Does the chart review assign severity or priority levels to its findings "
                "(e.g., high/medium/low, critical/minor, or a similar ranking system)?"
            ),
            points=1,
        )
    )

    rubric_items.append(
        RubricCategory(
            name="report_quality",
            description="Is the chart review report well-organized, professional, and suitable for a clinical audit?",
            failure="Disorganized or unprofessional format with no clinical reasoning.",
            minor_failure="Some structure but missing sections or lacking evidence-based rationale.",
            minor_success="Reasonably organized with clear sections, clinical language, and sound reasoning.",
            success="Professional audit report with structured findings, severity ratings, evidence citations, prioritized recommendations, and clear clinical rationale.",
            points=2,
        )
    )

    return RubricDatapoint(
        problem_statement=problem_statement,
        rubric=tuple(rubric_items),
        submission_instructions="Write your completed chart review to /testbed/chart_review.txt",
        available_tools=(
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
            tools.FINISH_TOOL,
        ),
        necessary_files={
            "/testbed/data/patient_chart.txt": chart_content,
            "/testbed/data/lab_results.csv": lab_content,
            "/testbed/data/medication_interactions.txt": interaction_content,
            "/testbed/data/clinical_guidelines.txt": guidelines_content,
        },
        problem_type="medical_chart_review",
    )
