#!/usr/bin/env python3
import os
import sys
from omegaconf import OmegaConf

def suggest_from_yaml_file(config_file, cur_word, prev_word):
    try:
        conf = OmegaConf.load(config_file)
        suggestions = get_suggestions(conf, cur_word, prev_word)
        print (" ".join(suggestions))
    except FileNotFoundError as e:
        sys.stderr.write(f"\n{e}\n")
        usage = "./pets/main.py -e cartpole"
        sys.stderr.write(
            f"Try run '{usage}' to get {config_file} for bash completion\n")

def select(conf, key):
    return conf if key == "" else conf.select(key)

def get_prefix(s):
    pos = s.rfind('.')
    return "" if pos == -1 else s[:pos]

def auto_complete_key(conf, key):
    sub_conf = conf.select(key)
    if hasattr(sub_conf, 'keys'):
        key += '.'
    else:
        key += f'=' # {sub_conf}'
    return key

def get_suggestions(conf, cur_word, prev_word=""):
    if cur_word == "=":
        value = select(conf, prev_word)
        if value is not None and not hasattr(value, 'keys'):
            return [f"{value}"]

    # Ex: "aaa.bbb" => "aaa"
    prefix = get_prefix(cur_word)

    # query subtree
    config = select(conf, prefix)

    # if there's nothing, return an empty list
    if not config or not hasattr(config, 'keys'):
        return []

    # Add dot (.) back to prefix if it's non-empty
    if prefix:
        prefix += '.'

    suggestions = [auto_complete_key(conf, prefix + key) for key in config]

    return suggestions 

def main():
    path_to_config = os.path.join(os.getcwd(), ".lastrun_config.yaml")
    cur_word = sys.argv[1]
    prev_word = sys.argv[2]
    suggest_from_yaml_file(path_to_config, cur_word, prev_word)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"\n{e}\n")
