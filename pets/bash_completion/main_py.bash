#!/usr/bin/env bash
_mbrl_pets_main_py ()
{ 
    # ref: `complete | grep find`
    local cur prev words cword;
    _init_completion || return;

    local ROOT=$(git rev-parse --show-toplevel)
    local helper=${ROOT}/pets/bash_completion/suggest.py
    local options=$([ -f $helper ] && $helper "$cur" "$prev")
    if [ "$cur" == "=" ]; then
        word=""
    else
        word=$cur
    fi
    COMPREPLY=($( compgen -W '$options' -- "$word" ));
}

# TODO(poweic): add bash-completion support to run main.py from anywhere
EXEC=./pets/main.py
complete -o nospace -F _mbrl_pets_main_py $EXEC
