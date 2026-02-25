import os
import select
import sys
import termios
import tty


def _check_env():
    if os.environ.get("KITTY_WINDOW_ID"):
        return True
    if os.environ.get("TERM_PROGRAM") in ("ghostty", "WezTerm"):
        return True
    return False


def supports_kitty() -> bool:
    if _check_env():
        return True
    if not sys.stdin.isatty():
        return False
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        os.write(sys.stdout.fileno(), b"\033_Gi=31,s=1,v=1,a=q,t=d,f=24;AAAA\033\\")
        response = b""
        for _ in range(20):
            if select.select([fd], [], [], 0.02)[0]:
                response += os.read(fd, 1024)
                if b"\\" in response:
                    break
            elif response:
                break
        return b"OK" in response
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


kitty_protocol_enabled = supports_kitty()
