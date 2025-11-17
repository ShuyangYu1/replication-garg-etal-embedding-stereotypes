import latexify
import numpy as np
import csv
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import random
import ast
import sys
import pylab
from utilities import *
from statsmodels.formula.api import ols
from statsmodels.iolib.summary2 import summary_col
import statsmodels.api as sm
from scipy.stats import kendalltau
import copy
import scipy
from scipy.stats.stats import pearsonr
from plot_creation import *
import traceback

pretty_axis_labels = {'male_pairs': 'Men', 'female_pairs': 'Women', 'names_asian': 'Asian', 'names_white': 'White', 'names_hispanic': 'Hispanic'}


def execute_plot_safely(plot_func, plot_args, plot_name="Unknown plot"):
    """
    Safely execute a plotting function with error handling.
    
    Args:
        plot_func: The plotting function to execute
        plot_args: Arguments to pass to the plotting function
        plot_name: Descriptive name of the plot for logging
    
    Returns:
        bool: True if successful, False if failed
    """
    try:
        print(f"\n{'='*60}")
        print(f"Generating: {plot_name}")
        print(f"Function: {plot_func.__name__}")
        print(f"{'='*60}")
        
        plot_func(*plot_args)
        
        print(f"✓ Successfully generated: {plot_name}")
        return True
        
    except Exception as e:
        print(f"\n{'!'*60}")
        print(f"✗ ERROR generating: {plot_name}")
        print(f"Function: {plot_func.__name__}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"\nFull traceback:")
        print(traceback.format_exc())
        print(f"{'!'*60}\n")
        return False


def execute_plot_list(plot_list, category_name=""):
    """
    Execute a list of plots with error tracking.
    
    Args:
        plot_list: List of [function, args] pairs
        category_name: Name of the plot category for logging
    
    Returns:
        tuple: (successful_count, failed_count, failed_plots)
    """
    successful = 0
    failed = 0
    failed_plots = []
    
    print(f"\n{'#'*60}")
    print(f"Starting category: {category_name}")
    print(f"Total plots in category: {len(plot_list)}")
    print(f"{'#'*60}\n")
    
    for idx, plot in enumerate(plot_list, 1):
        plot_func = plot[0]
        plot_args = plot[1]
        plot_name = f"{category_name} - Plot {idx}/{len(plot_list)}"
        
        success = execute_plot_safely(plot_func, plot_args, plot_name)
        
        if success:
            successful += 1
        else:
            failed += 1
            failed_plots.append({
                'category': category_name,
                'plot_number': idx,
                'function': plot_func.__name__,
                'args_summary': str(plot_args[1:])[:100]  # First 100 chars of args
            })
    
    print(f"\n{'='*60}")
    print(f"Category '{category_name}' complete:")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"{'='*60}\n")
    
    return successful, failed, failed_plots


def main(filenametodo='run_results/finalrun.csv'):
    plots_folder = 'plots/'
    set_plots_folder(plots_folder)

    print("Loading data file...")
    try:
        rows = load_file(filenametodo)
        print(f"✓ Data loaded successfully")
        print(f"Available keys: {list(rows.keys())}")
    except Exception as e:
        print(f"✗ CRITICAL ERROR: Could not load data file: {filenametodo}")
        print(f"Error: {e}")
        return

    # Define all plot configurations
    plots_to_do_gender_static = [
        [scatter_occupation_percents_distances, [rows['google'], 'google', 'occupations1950', 'male_pairs', 'female_pairs', -1, 'data/occupation_percentages_gender_occ1950.csv', load_occupationpercent_data, occupation_func_female_percent, [-.15, .15], [-100, 100], False, False, 'norm', 'png']],
        [scatter_occupation_percents_distances, [rows['google'], 'google', 'occupations1950', 'male_pairs', 'female_pairs', -1, 'data/occupation_percentages_gender_occ1950.csv', load_occupationpercent_data, occupation_func_female_logitprop, [-.15, .15], [-5, 3], False, False, 'norm', 'pdf']],
        [residual_analysis_with_stereotypes, [rows['google'], 'google', 'occupations1950', 'male_pairs', 'female_pairs', 'data/occupation_percentages_gender_occ1950.csv', load_occupationpercent_data, occupation_func_female_percent, 'data/mturk_stereotypes.csv', load_mturkstereotype_data, 'norm', 'pdf']],
    ]

    plots_to_do_gender_dynamic = [
        [plot_overtime_scatter, [rows['sgns'], 'sgns', 'occupations1950', 'male_pairs', 'female_pairs', 'data/occupation_percentages_gender_occ1950.csv', occupation_func_female_percent, None, None, False, None, None]],
        [plot_averagebias_over_time_consistentoccupations, [rows['sgns'], 'sgns', 'occupations1950', 'male_pairs', 'female_pairs', True, 'data/occupation_percentages_gender_occ1950.csv', occupation_func_female_logitprop, 0, False, '', None, None, False]],
        [plot_averagebias_over_time_consistentoccupations, [rows['sgns'], 'sgns', 'occupations1950', 'male_pairs', 'female_pairs', True, 'data/occupation_percentages_gender_occ1950.csv', occupation_func_female_percent, 0, False, '', None, None, False]],
        [create_cross_time_correlation_heatmap_differencestoself, [rows['sgns'], 'sgns', 'personalitytraits_original', 'male_pairs', 'female_pairs', None, 'pdf']],
        [create_cross_time_correlation_heatmap_differencestoself, [rows['sgns'], 'sgns', 'occupations1950', 'male_pairs', 'female_pairs', None, 'pdf']],
    ]

    plots_to_do_race_dynamic = [
        [create_cross_time_correlation_heatmap_differencestoself, [rows['sgns'], 'sgns', 'personalitytraits_original', 'names_white', 'names_hispanic', None, 'pdf']],
        [create_cross_time_correlation_heatmap_differencestoself, [rows['sgns'], 'sgns', 'personalitytraits_original', 'names_white', 'names_asian']],
        [create_cross_time_correlation_heatmap_differencestoself, [rows['sgns'], 'sgns', 'personalitytraits_original', 'names_white', 'names_russian']],
        [plot_averagebias_over_time_consistentoccupations, [rows['sgns'], 'sgns', 'occupations1950', 'names_white', 'names_asian', True, 'data/occupation_percentages_race_occ1950.csv', occupation_func_whiteasian_logitprop, 0, False, '', None, None, False]],
        [plot_averagebias_over_time_consistentoccupations, [rows['sgns'], 'sgns', 'occupations1950', 'names_white', 'names_asian', True, 'data/occupation_percentages_race_occ1950.csv', occupation_func_whiteasian_percent, 0, False, '', None, None, False]],
        [plot_averagebias_over_time_consistentoccupations, [rows['sgns'], 'sgns', 'occupations1950', 'names_white', 'names_hispanic', True, 'data/occupation_percentages_race_occ1950.csv', occupation_func_whitehispanic_logitprop, 0, False, '', None, None, False]],
        [plot_averagebias_over_time_consistentoccupations, [rows['sgns'], 'sgns', 'occupations1950', 'names_white', 'names_hispanic', True, 'data/occupation_percentages_race_occ1950.csv', occupation_func_whitehispanic_percent, 0, False, '', None, None, False]],
        [plot_averagebias_over_time_consistentoccupations, [rows['sgns'], 'sgns', 'adjectives_otherization', 'names_white', 'names_asian', False]],
    ]

    plots_to_do_appendix_general = [
        [plot_mean_counts_together, [rows['sgns'], 'sgns', ['names_chinese', 'names_white', 'names_asian', 'names_hispanic', 'names_russian', 'male_pairs', 'female_pairs'], 'groups']],
        [plot_vector_variances_together, [rows['sgns'], 'sgns', ['names_chinese', 'names_white', 'names_asian', 'names_hispanic', 'names_russian', 'male_pairs', 'female_pairs'], 'groups']],
        [plot_mean_counts_together, [rows['sgns'], 'sgns', ['adjectives_princeton', 'adjectives_otherization', 'personalitytraits_original', 'occupations1950', 'adjectives_williamsbest', 'adjectives_appearance', 'adjectives_intelligencegeneral'], 'neutrals']],
        [plot_vector_variances_together, [rows['sgns'], 'sgns', ['adjectives_princeton', 'adjectives_otherization', 'personalitytraits_original', 'occupations1950', 'adjectives_williamsbest', 'adjectives_appearance', 'adjectives_intelligencegeneral'], 'neutrals']],
    ]

    plots_to_do_appendix_gender_static = [
        [scatter_occupation_percents_distances, [rows['google'], 'google', 'occupations1950_professional', 'male_pairs', 'female_pairs', -1, 'data/occupation_percentages_gender_occ1950.csv', load_occupationpercent_data, occupation_func_female_percent, [-.15, .15], [-100, 100], False, False, 'norm', 'pdf']],
        [scatter_occupation_percents_distances, [rows['google'], 'google', 'occupations1950_professional', 'male_pairs', 'female_pairs', -1, 'data/occupation_percentages_gender_occ1950.csv', load_occupationpercent_data, occupation_func_female_logitprop, [-.15, .15], [-5, 3], False, False, 'norm', 'pdf']],
        [scatter_occupation_percents_distances, [rows['sgns'], 'sgns', 'adjectives_williamsbest', 'male_pairs', 'female_pairs', -1, 'data/adjectives_williamsbest.csv', load_williamsbestadjectives, occupation_func_williamsbestadject, None, [-500, 500], True, False, 'norm']],
        [scatter_occupation_percents_distances, [rows['sgns'], 'sgns', 'adjectives_williamsbest', 'male_pairs', 'female_pairs', -3, 'data/adjectives_williamsbest.csv', load_williamsbestadjectives, occupation_func_williamsbestadject, None, [-500, 500], True, False, 'norm']],
    ]

    plots_to_do_appendix_raceasian_static = [
        [princeton_trilogy_plots, [rows['sgns'], 'sgns', 'names_white', 'names_chinese', 'chinese']],
        [scatter_occupation_percents_distances, [rows['google'], 'google', 'occupations1950', 'names_white', 'names_asian', -1, 'data/occupation_percentages_race_occ1950.csv', load_occupationpercent_data, occupation_func_whiteasian_logitprop, None, None, False, False, 'norm', 'pdf']],
        [scatter_occupation_percents_distances, [rows['google'], 'google', 'occupations1950', 'names_white', 'names_asian', -1, 'data/occupation_percentages_race_occ1950.csv', load_occupationpercent_data, occupation_func_whiteasian_percent, None, None, False, False, 'norm', 'pdf']],
    ]

    plots_to_do_appendix_racehispanic_static = [
        [scatter_occupation_percents_distances, [rows['google'], 'google', 'occupations1950', 'names_white', 'names_hispanic', -1, 'data/occupation_percentages_race_occ1950.csv', load_occupationpercent_data, occupation_func_whitehispanic_logitprop, None, None, False, False, 'norm', 'pdf']],
        [scatter_occupation_percents_distances, [rows['google'], 'google', 'occupations1950', 'names_white', 'names_hispanic', -1, 'data/occupation_percentages_race_occ1950.csv', load_occupationpercent_data, occupation_func_whitehispanic_percent, None, None, False, False, 'norm', 'pdf']],
    ]

    plots_to_do_appendix_gender_dynamic = [
        [do_over_time_trend_test, [rows['sgns'], 'sgns', 'adjectives_intelligencegeneral', 'male_pairs', 'female_pairs', False, '', range(1960, 2000, 10)]],
        [do_over_time_trend_test, [rows['sgns'], 'sgns', 'adjectives_appearance', 'male_pairs', 'female_pairs', False, '', range(1960, 2000, 10)]],
    ]

    # Track overall statistics
    all_failed_plots = []
    total_successful = 0
    total_failed = 0

    # Execute all plot categories
    plot_categories = [
        (plots_folder + 'gender/', [
            ('Gender Static', plots_to_do_gender_static),
            ('Gender Dynamic', plots_to_do_gender_dynamic)
        ]),
        (plots_folder + 'ethnicity/', [
            ('Race Dynamic', plots_to_do_race_dynamic)
        ]),
        (plots_folder + 'appendix/', [
            ('Appendix General', plots_to_do_appendix_general)
        ]),
        (plots_folder + 'appendix/gender/', [
            ('Appendix Gender Static', plots_to_do_appendix_gender_static),
            ('Appendix Gender Dynamic', plots_to_do_appendix_gender_dynamic)
        ]),
        (plots_folder + 'appendix/ethnicity/', [
            ('Appendix Race Asian Static', plots_to_do_appendix_raceasian_static),
            ('Appendix Race Hispanic Static', plots_to_do_appendix_racehispanic_static)
        ])
    ]

    for folder_path, categories in plot_categories:
        set_plots_folder(folder_path)
        
        for category_name, plot_list in categories:
            successful, failed, failed_list = execute_plot_list(plot_list, category_name)
            total_successful += successful
            total_failed += failed
            all_failed_plots.extend(failed_list)

    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total plots attempted: {total_successful + total_failed}")
    print(f"✓ Successfully generated: {total_successful}")
    print(f"✗ Failed: {total_failed}")
    print(f"Success rate: {100 * total_successful / (total_successful + total_failed):.1f}%")
    
    if all_failed_plots:
        print(f"\n{'='*60}")
        print("FAILED PLOTS DETAILS:")
        print("="*60)
        for i, plot_info in enumerate(all_failed_plots, 1):
            print(f"\n{i}. {plot_info['category']} - Plot #{plot_info['plot_number']}")
            print(f"   Function: {plot_info['function']}")
            print(f"   Args: {plot_info['args_summary']}")
    else:
        print("\n✓ All plots generated successfully!")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
