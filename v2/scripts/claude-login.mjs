/**
 * Make sure the person running JARVIS is logged in to *their own* Claude
 * account before the brain starts.
 *
 * JARVIS has no account of its own and the repository carries no credentials.
 * The brain runs on whatever Claude Code login exists on this computer, so
 * everyone who downloads it uses their own subscription, their own MCP servers
 * and their own history, and nobody's account travels with the code.
 *
 * Without this check, a fresh download with no login looked alive (the reactor
 * spun, the microphone heard you) and then answered nothing. Now it stops at
 * the door and walks them through signing in.
 */

import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import process from 'node:process'

const isWindows = process.platform === 'win32'

/**
 * Where `claude` actually is. On Windows, npm's global folder is often missing
 * from PATH right after installing, which makes `claude` "not recognized" even
 * though it installed fine, so look there directly as well.
 */
function findClaude() {
  const probe = spawnSync('claude', ['--version'], { shell: isWindows, encoding: 'utf8' })
  if (probe.status === 0) return 'claude'
  if (isWindows && process.env.APPDATA) {
    const npmBin = join(process.env.APPDATA, 'npm', 'claude.cmd')
    if (existsSync(npmBin)) return `"${npmBin}"`
  }
  return null
}

/** true / false, or null when this version of Claude Code can't say. */
function loggedIn(claude) {
  const res = spawnSync(claude, ['auth', 'status', '--json'], {
    shell: isWindows,
    encoding: 'utf8',
  })
  try {
    return JSON.parse(res.stdout).loggedIn === true
  } catch {
    return null
  }
}

/**
 * Returns true when it is fine to start the brain. Prints plain instructions
 * and returns false when it is not.
 */
export function ensureClaudeLogin() {
  // Someone who has deliberately set up a key or token has made their choice.
  if (process.env.ANTHROPIC_API_KEY || process.env.CLAUDE_CODE_OAUTH_TOKEN) return true

  const claude = findClaude()
  if (!claude) {
    console.log(
      '\n  Claude Code is not installed on this computer.\n' +
        '  JARVIS runs on your own Claude account, through Claude Code. Install it with:\n\n' +
        '      npm install -g @anthropic-ai/claude-code\n\n' +
        '  then close this window, open a new one, and run `npm start` again.\n',
    )
    return false
  }

  const state = loggedIn(claude)
  if (state === true) {
    console.log('  Claude account: logged in on this computer.')
    return true
  }
  if (state === null) {
    // An older Claude Code without `auth status`. Carry on; if it is not logged
    // in, the brain says so on screen.
    console.log('  Could not check the Claude login (update Claude Code to enable the check).')
    return true
  }

  console.log(
    '\n  JARVIS needs a Claude account, and this computer is not logged in to one.\n' +
      '  Sign in with YOUR OWN account. Your browser will open to do it.\n',
  )
  const login = spawnSync(claude, ['auth', 'login'], { shell: isWindows, stdio: 'inherit' })
  if (login.status === 0 && loggedIn(claude) === true) {
    console.log('\n  Logged in. Starting JARVIS.\n')
    return true
  }
  console.log(
    '\n  Login did not finish. Run `claude` in this window, choose to sign in,\n' +
      '  then run `npm start` again.\n',
  )
  return false
}
