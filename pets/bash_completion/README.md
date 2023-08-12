# Bash-completion

## Getting Started
Bash-completion let you do awesome things like this:
```bash
$ find -m<TAB><TAB>
-maxdepth  -mindepth  -mmin      -mount     -mtime

$ find -maxdepth<TAB><TAB>
0  1  2  3  4  5  6  7  8  9
```
(if you're in ubuntu and don't have bash-completion, run
`apt-get install bash-completion` to install the package.)

To activate it, simply run `source pets/bash_completion/main_py.bash` or put
this file in directory `/etc/bash_completion.d/`.

```bash
$ ./pets/main.py<TAB><TAB>
checkpoint=       env.              log_dir=          policy.           training.
device=           full_log_dir=     motor_babbling.   presets.          trial_timesteps=
dynamics_model.   log_config=       num_trials=       random_seed=

$ ./pets/main.py tr<TAB><TAB>
training.         trial_timesteps=

$ ./pets/main.py training.<TAB><TAB>
training.batch_size=          training.incremental=         training.optimizer.
training.full_epochs=         training.incremental_epochs=  training.testing.

$ ./pets/main.py training.optimizer.<TAB><TAB>
training.optimizer.clazz=   training.optimizer.params.

$ ./pets/main.py training.optimizer.params.l<TAB><TAB>

$ ./pets/main.py training.optimizer.params.lr=
```

## How It Is Done

With `complete`, `compgen`, and some bash programming, we can do the same for
`pets/main.py`. Fortunately, we don't have to do all of these by ourself.
`complete` keeps all the bash completion scripts for commands like `find`, `tar`
, `ssh`, etc. We can simply reuse some of the functions and logics in there.
```bash
$ find -m<TAB><TAB>      # trigger bash completion for `find`

$ complete | grep "find" # list all registered bash completion see what is used by find
complete -F _find find   # command `find` is bind to a bash function _find through `-F` argument

$ type _find             # get the function defintion of _find
_find is a function
_find ()
{
    local cur prev words cword;
    _init_completion || return;
    case $prev in
        -maxdepth | -mindepth)
            COMPREPLY=($( compgen -W '{0..9}' -- "$cur" ));
            return 0
        ;;

    # 80 more lines here ...

            for j in ${!COMPREPLY[@]};
            do
                [[ ${COMPREPLY[j]} == $i ]] && unset COMPREPLY[j];
            done;
        done;
    fi;
    _filedir;
    return 0
}
```

Together with a python script that loads a YAML config file last used by
`pets/main.py` and generates suggestions, we can have our own bash completion.
