"""Shell completion scripts. Each asks `scramjit __complete WORDS...` for candidates, so completion
always matches the installed command tree; a last line `__file__` or `__dir__` asks the shell
to complete paths."""

BASH = r'''# scramjit completion for bash: add to ~/.bashrc, or: source <(scramjit completion bash)
_scramjit() {
    local cur=${COMP_WORDS[COMP_CWORD]} IFS=$'\n' kind=
    local out=($(scramjit __complete "${COMP_WORDS[@]:1:COMP_CWORD}" 2>/dev/null))
    if (( ${#out[@]} )) && [[ ${out[${#out[@]}-1]} == __*__ ]]; then
        kind=${out[${#out[@]}-1]}
        unset "out[${#out[@]}-1]"
    fi
    COMPREPLY=($(compgen -W "${out[*]}" -- "$cur"))
    case $kind in
        __file__) COMPREPLY+=($(compgen -f -- "$cur")) ;;
        __dir__) COMPREPLY+=($(compgen -d -- "$cur")) ;;
    esac
}
complete -o filenames -o bashdefault -F _scramjit scramjit
'''

ZSH = r'''#compdef scramjit
# scramjit completion for zsh: save as _scramjit in a directory on $fpath, or: source <(scramjit completion zsh)
_scramjit() {
    local -a out
    local kind
    out=("${(@f)$(scramjit __complete "${words[@]:1:$((CURRENT-1))}" 2>/dev/null)}")
    out=(${out:#})
    if [[ ${out[-1]} == __*__ ]]; then
        kind=${out[-1]}
        out=("${(@)out[1,-2]}")
    fi
    (( ${#out} )) && compadd -a out
    case $kind in
        __file__) _files ;;
        __dir__) _files -/ ;;
    esac
}
if [[ $zsh_eval_context[-1] == loadautofunc ]]; then
    _scramjit "$@"
else
    compdef _scramjit scramjit
fi
'''

FISH = r'''# scramjit completion for fish: save as ~/.config/fish/completions/scramjit.fish
function __scramjit_complete
    set -l words (commandline -opc)
    set -e words[1]
    set -l cur (commandline -ct)
    for line in (scramjit __complete $words "$cur" 2>/dev/null)
        switch $line
            case __file__
                __fish_complete_path "$cur"
            case __dir__
                __fish_complete_directories "$cur"
            case '*'
                echo $line
        end
    end
end
complete -c scramjit -f -a '(__scramjit_complete)'
'''
