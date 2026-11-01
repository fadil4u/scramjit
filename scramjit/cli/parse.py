"""A small declarative command-line framework: nested commands, options and arguments, help at
every level, "did you mean" suggestions and shell completion. Standard library only.

    root = Command('scramjit', 'Python at native speed')
    p = root.command('preflight', 'Show what compiles', func=do_preflight)
    p.arg('paths', 'Files or directories', nargs='*')
    p.opt('-q', '--quiet', flag=True, help='...')
"""

import difflib

from .term import wrap


class UsageError(Exception):
    """A command-line mistake: printed with the command's usage and a hint, exit status 2."""

    def __init__(self, message, command=None, suggestions=(), hint=None):
        Exception.__init__(self, message)
        self.message, self.command = message, command
        self.suggestions, self.hint = list(suggestions), hint


class Opt:
    """An option. flag=True: takes no value. multiple=True: may repeat, collected in a list.
    choices: allowed values. complete: 'file', 'dir' or None, for shell completion."""

    def __init__(self, *flags, dest=None, metavar=None, help='', flag=False, multiple=False,
                 choices=None, default=None, type=str, group=None, hidden=False, complete=None,
                 const=True):
        self.flags = flags
        self.long = next((f for f in flags if f.startswith('--')), None)
        self.short = next((f for f in flags if not f.startswith('--')), None)
        self.dest = dest or (self.long or self.short).lstrip('-').replace('-', '_')
        self.metavar = metavar or self.dest
        self.help, self.flag, self.multiple = help, flag, multiple
        self.choices, self.type, self.group, self.hidden = choices, type, group, hidden
        self.complete, self.const = complete, const
        self.default = default if default is not None else ([] if multiple else
                                                             (False if flag else None))

    def display(self, term):
        names = ', '.join(term.style(f, 'option') for f in self.flags)
        if not self.long:
            names = names
        elif not self.short:
            names = '    ' + names
        if not self.flag:
            names += ' ' + term.style('<%s>' % self.metavar, 'metavar')
        return names

    def convert(self, value, command):
        if self.choices and value not in self.choices:
            raise UsageError("invalid value '%s' for %s" % (value, self.long or self.short),
                             command, _close(value, self.choices),
                             hint='choose from: %s' % ', '.join(self.choices))
        try:
            return self.type(value)
        except ValueError:
            raise UsageError("invalid value '%s' for %s" % (value, self.long or self.short),
                             command) from None


class Arg:
    """A positional argument. nargs: 1, '?' (optional), '*' (any number) or 'rest' (this and
    everything after it, options included: arguments passed through to something else)."""

    def __init__(self, name, help='', nargs=1, metavar=None, complete='file'):
        self.name, self.help, self.nargs = name, help, nargs
        self.metavar = metavar or name
        self.complete = complete

    def display(self, term):
        m = self.metavar
        if self.nargs == 1:
            return term.style('<%s>' % m, 'metavar')
        if self.nargs == '?':
            return term.style('[%s]' % m, 'metavar')
        return term.style('[%s...]' % m, 'metavar')


class Command:
    def __init__(self, name, summary, description=None, func=None, parent=None, hidden=False,
                 aliases=(), examples=()):
        self.name, self.summary, self.description = name, summary, description
        self.func, self.parent, self.hidden = func, parent, hidden
        self.aliases, self.examples = tuple(aliases), list(examples)
        self.options, self.args, self.commands = [], [], {}
        self.epilog = None

    # building

    def command(self, name, summary, **kw):
        c = Command(name, summary, parent=self, **kw)
        self.commands[name] = c
        return c

    def opt(self, *flags, **kw):
        self.options.append(Opt(*flags, **kw))
        return self

    def arg(self, name, help='', **kw):
        self.args.append(Arg(name, help, **kw))
        return self

    def example(self, cmdline, comment=None):
        self.examples.append((cmdline, comment))
        return self

    # lookup

    @property
    def path(self):
        parts, c = [], self
        while c:
            parts.append(c.name)
            c = c.parent
        return ' '.join(reversed(parts))

    @property
    def root(self):
        c = self
        while c.parent:
            c = c.parent
        return c

    def visible_commands(self):
        return [c for c in self.commands.values() if not c.hidden]

    def find_command(self, word):
        if word in self.commands:
            return self.commands[word]
        for c in self.commands.values():
            if word in c.aliases:
                return c
        return None

    def all_options(self):
        """This command's options, then the global (root) options."""
        opts = list(self.options)
        if self.parent:
            opts += [o for o in self.root.options if o not in opts]
        return opts

    def find_option(self, flag):
        for o in self.all_options():
            if flag in o.flags:
                return o
        return None


def _close(word, candidates):
    return difflib.get_close_matches(word, list(candidates), n=3, cutoff=0.6)


# ---------------------------------------------------------------------------------------------
# parsing

class Parsed:
    def __init__(self, command, values, extra=None):
        self.command, self.values = command, values

    def __getattr__(self, name):
        try:
            return self.values[name]
        except KeyError:
            raise AttributeError(name) from None


def parse(root, argv):
    """argv (without the program name) -> Parsed. Raises UsageError. `--help` anywhere sets
    values['help']; the command path is resolved as far as it goes first."""
    cmd = root
    values = {}
    for o in root.options:
        values[o.dest] = list(o.default) if o.multiple else o.default
    positionals = []
    i, n = 0, len(argv)
    only_args = False
    while i < n:
        a = argv[i]
        rest = _rest_arg(cmd, positionals)
        if rest is not None and len(positionals) >= _fixed(cmd):
            positionals.extend(argv[i:])
            break
        if only_args or a == '-' or not a.startswith('-'):
            if not positionals and cmd.commands and not only_args:
                sub = cmd.find_command(a)
                if sub is None:
                    raise UsageError(
                        "unknown command '%s'" % a, cmd,
                        _close(a, [c.name for c in cmd.visible_commands()]),
                        hint=_file_hint(cmd, a))
                cmd = sub
                for o in cmd.options:
                    values.setdefault(o.dest, list(o.default) if o.multiple else o.default)
                i += 1
                continue
            positionals.append(a)
            i += 1
            continue
        if a == '--':
            only_args = True
            i += 1
            continue
        if a.startswith('--'):
            name, eq, val = a.partition('=')
            o = cmd.find_option(name)
            if o is None:
                raise UsageError("unknown option '%s'" % name, cmd,
                                 _close(name, [f for x in cmd.all_options() if not x.hidden
                                               for f in x.flags if f.startswith('--')]))
            if o.flag:
                if eq:
                    raise UsageError("option '%s' takes no value" % name, cmd)
                _set(values, o, o.const)
            else:
                if not eq:
                    i += 1
                    if i >= n:
                        raise UsageError("option '%s' needs a value <%s>" % (name, o.metavar), cmd)
                    val = argv[i]
                _set(values, o, o.convert(val, cmd))
            i += 1
            continue
        o = cmd.find_option(a)  # a whole short option of several letters, like -O0
        if o is not None and o.flag:
            _set(values, o, o.const)
            i += 1
            continue
        # short options: -I dir, -Idir, -qv
        j = 1
        while j < len(a):
            flag = '-' + a[j]
            o = cmd.find_option(flag)
            if o is None:
                raise UsageError("unknown option '%s'" % flag, cmd,
                                 _close(a, [f for x in cmd.all_options() if not x.hidden
                                            for f in x.flags]))
            if o.flag:
                _set(values, o, o.const)
                j += 1
                continue
            val = a[j + 1:]
            if not val:
                i += 1
                if i >= n:
                    raise UsageError("option '%s' needs a value <%s>" % (flag, o.metavar), cmd)
                val = argv[i]
            _set(values, o, o.convert(val, cmd))
            break
        i += 1
    if values.get('help'):
        return Parsed(cmd, values)
    _bind(cmd, positionals, values)
    return Parsed(cmd, values)


def _set(values, o, v):
    if o.multiple:
        values.setdefault(o.dest, []).append(v)
    else:
        values[o.dest] = v


def _fixed(cmd):
    return sum(1 for a in cmd.args if a.nargs == 1)


def _rest_arg(cmd, positionals):
    for a in cmd.args:
        if a.nargs == 'rest':
            return a
    return None


def _bind(cmd, positionals, values):
    if cmd.commands and not cmd.args:
        if positionals:
            raise UsageError("unexpected argument '%s'" % positionals[0], cmd)
        return
    pos = list(positionals)
    for k, a in enumerate(cmd.args):
        if a.nargs == 1:
            if not pos:
                raise UsageError('missing argument <%s>' % a.metavar, cmd)
            values[a.name] = pos.pop(0)
        elif a.nargs == '?':
            values[a.name] = pos.pop(0) if pos else None
        else:
            values[a.name], pos = pos, []
    if pos:
        raise UsageError("unexpected argument '%s'" % pos[0], cmd)


def _file_hint(cmd, word):
    if cmd.parent is None and 'preflight' in cmd.commands and (word.endswith('.py') or ':' in word or '/' in word):
        return "to check it: `%s preflight %s`" % (cmd.name, word)
    return None


# ---------------------------------------------------------------------------------------------
# help

def usage_line(cmd, term):
    parts = [term.style(cmd.path, 'command')]
    if cmd.commands:
        parts.append(term.style('<command>', 'metavar'))
    if [o for o in cmd.all_options() if not o.hidden]:
        parts.append(term.style('[options]', 'metavar'))
    parts += [a.display(term) for a in cmd.args]
    return '%s %s' % (term.style('Usage:', 'heading'), ' '.join(parts))


def render_help(cmd, term):
    w = term.width
    out = [term.style(cmd.summary, 'bold')]
    if cmd.description:
        out.append('')
        for para in cmd.description.strip().split('\n\n'):
            out.extend(wrap(term.code(' '.join(para.split())), w))
            out.append('')
        out.pop()
    out.append('')
    out.append(usage_line(cmd, term))

    sections = [('Commands:', [(term.style(c.name, 'command'), c.summary)
                               for c in cmd.visible_commands()]),
                ('Arguments:', [(a.display(term), a.help) for a in cmd.args if a.help])]
    groups = {'Options:': []}
    for o in cmd.options:
        if not o.hidden:
            groups.setdefault(o.group or 'Options:', []).append(o)
    sections += [(t, [(o.display(term), _opt_help(o)) for o in opts])
                 for t, opts in groups.items()]
    if cmd.parent is not None:
        sections.append(('Global options:', [(o.display(term), _opt_help(o))
                                             for o in cmd.root.options
                                             if not o.hidden and o.dest != 'version']))
    rows = [r for _, rs in sections for r in rs]
    col = min(max([_plain_len(term, r[0]) for r in rows] or [0]) + 4, 34)
    for title, rs in sections:
        if not rs:
            continue
        out.append('')
        out.append(term.style(title, 'heading'))
        for left, right in rs:
            pad = col - _plain_len(term, left) - 2
            if pad < 2:
                out.append('  ' + left)
                out.extend(wrap(term.code(right), w, indent=col))
            else:
                lines = wrap(term.code(right), w, indent=col, first=col)
                out.append('  ' + left + ' ' * pad + lines[0].lstrip())
                out.extend(lines[1:])
    if cmd.examples:
        out.append('')
        out.append(term.style('Examples:', 'heading'))
        for line, comment in cmd.examples:
            if comment:
                out.append('  ' + term.style('# ' + comment, 'muted'))
            out.append('  %s %s' % (term.style('$', 'prompt'), line))
    if cmd.epilog:
        out.append('')
        out.extend(wrap(term.code(cmd.epilog), w))
    if cmd.commands:
        out.append('')
        out.append(term.code('Run `%s <command> --help` for more on a command.' % cmd.path))
    return '\n'.join(out)


def _opt_help(o):
    h = o.help
    if o.choices and '`' not in h:
        h += ' (%s)' % ', '.join('`%s`' % c for c in o.choices)
    if o.multiple:
        h += ' (repeatable)'
    return h


def _plain(term, s):
    from .term import _ANSI
    return _ANSI.sub('', s)


def _plain_len(term, s):
    return len(_plain(term, s))


# ---------------------------------------------------------------------------------------------
# completion

def complete(root, words):
    """Candidates for the last word of `words` (the command line after the program name, the
    last item being the word under the cursor, possibly empty). Returns (candidates, kind):
    kind 'file' or 'dir' asks the shell to complete paths."""
    cmd = root
    cur = words[-1] if words else ''
    prev = words[:-1]
    expect = None
    npos = 0
    for k, w in enumerate(prev):
        if expect is not None:
            expect = None
            continue
        if w.startswith('-') and w != '-':
            o = cmd.find_option(w.partition('=')[0])
            if (o is not None and not o.flag and '=' not in w
                    and (w.startswith('--') or len(w) == 2)):
                expect = o
            continue
        sub = cmd.find_command(w) if npos == 0 else None
        if sub is not None:
            cmd = sub
            continue
        npos += 1
    if expect is not None:
        if expect.choices:
            return [c for c in expect.choices if c.startswith(cur)], None
        return [], expect.complete
    if cur.startswith('-'):
        flags = [f for o in cmd.all_options()
                 if not o.hidden and (cmd is root or o.dest != 'version')
                 for f in o.flags if f.startswith('--') or len(cur) <= 2]
        return sorted(f for f in flags if f.startswith(cur)), None
    if cmd.commands and npos == 0:
        return sorted(c.name for c in cmd.visible_commands() if c.name.startswith(cur)), None
    for a in cmd.args:
        return [], a.complete
    return [], None
