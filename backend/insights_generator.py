def generate_insights(phoneme_scores):
    insights = []

    # Track improvements
    max_improvement = 0
    from_session, to_session = 0, 0

    for i in range(len(phoneme_scores)):
        for j in range(i + 1, len(phoneme_scores)):
            improvement = phoneme_scores[j] - phoneme_scores[i]
            if improvement > max_improvement:
                max_improvement = improvement
                from_session = i + 1
                to_session = j + 1

    if max_improvement > 0:
        insights.append({
            'type': 'improvement',
            'icon': '✅',
            'message': f"Your phoneme clarity improved by {max_improvement:.2f}% from Session {from_session} to Session {to_session}."
        })

    # Spot lowest score
    min_score = min(phoneme_scores)
    min_index = phoneme_scores.index(min_score) + 1

    if min_score < 30:  # Threshold for bad performance
        insights.append({
            'type': 'warning',
            'icon': '⚠️',
            'message': f"Session {min_index} needs review: phoneme accuracy dropped to {min_score:.2f}%."
        })

    # Highlight best session
    max_score = max(phoneme_scores)
    max_index = phoneme_scores.index(max_score) + 1
    insights.append({
        'type': 'achievement',
        'icon': '🏆',
        'message': f"Best phoneme performance in Session {max_index} with {max_score:.2f}% accuracy."
    })

    return insights

def generate_session_insights(results):
    """Generate insights from session results."""
    try:
        # Extract phoneme scores with proper string handling
        phoneme_scores = []
        for idx, result in enumerate(results):
            try:
                score = result['metrics']['text_comparison']['phoneme_correctness']
                # Handle the percentage string properly
                if isinstance(score, str):
                    # Remove the % sign and convert to float
                    score = float(score.replace('%', ''))
                else:
                    score = float(score)
                phoneme_scores.append(score)
                print(f"Successfully processed session {idx + 1}: {score}%")
            except (KeyError, ValueError, TypeError) as e:
                print(f"Error processing session {idx + 1}: {str(e)}")
                continue

        print("Extracted phoneme scores:", phoneme_scores)

        if not phoneme_scores:
            return [{
                'type': 'info',
                'icon': 'ℹ️',
                'message': 'No valid session data available for analysis.'
            }]

        # Generate insights using the provided function
        insights = generate_insights(phoneme_scores)
        print("Generated insights:", insights)
        return insights

    except Exception as e:
        print(f"Error in generate_session_insights: {str(e)}")
        return [{
            'type': 'error',
            'icon': '❌',
            'message': str(e)
        }] 