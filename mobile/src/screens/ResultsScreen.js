import React from 'react';
import { View, Text, ScrollView, StyleSheet } from 'react-native';
import { useRoute } from '@react-navigation/native';

const Section = ({ title, children }) => (
	<View style={styles.section}>
		<Text style={styles.sectionTitle}>{title}</Text>
		{children}
	</View>
);

const AdviceItem = ({ text, type = 'default' }) => (
	<View style={[styles.adviceItem, styles[`advice_${type}`]]}>
		<Text style={styles.adviceText}>{text}</Text>
	</View>
);

export default function ResultsScreen() {
	const route = useRoute();
	const result = route.params?.result;

	if (!result) {
		return (
			<View style={styles.emptyContainer}>
				<Text style={styles.emptyText}>No analysis yet. Upload a video from the Analyze tab.</Text>
			</View>
		);
	}

	const renderPairs = (obj) => {
		return Object.entries(obj).map(([k, v]) => (
			<Text key={k} style={styles.dataText}>{k}: {String(v)}</Text>
		));
	};

	const renderSimilarity = (similarity) => {
		if (!similarity) return <Text style={styles.dataText}>No similarity available.</Text>;
		const keys = Object.keys(similarity);
		if (keys.length === 0) return <Text style={styles.dataText}>No similarity available.</Text>;
		return keys.map((k) => (
			<Text key={k} style={styles.dataText}>{k}: {(similarity[k] * 100).toFixed(0)}%</Text>
		));
	};

	const renderAdviceList = (list, type = 'default') => {
		if (!list || !Array.isArray(list)) return null;
		return list.map((item, index) => (
			<AdviceItem key={index} text={item} type={type} />
		));
	};

	return (
		<ScrollView contentContainerStyle={styles.container}>
			<Text style={styles.mainTitle}>Analysis Results</Text>
			<Text style={styles.filename}>{result.filename}</Text>

			{/* AI Coaching Advice */}
			{result.coaching_advice && (
				<Section title="🤖 AI Coach Advice">
					{result.coaching_advice.summary && (
						<View style={styles.summaryContainer}>
							<Text style={styles.summaryText}>{result.coaching_advice.summary}</Text>
						</View>
					)}
					
					{result.coaching_advice.priority_issues && (
						<View style={styles.adviceSection}>
							<Text style={styles.adviceSectionTitle}>Priority Issues to Fix:</Text>
							{renderAdviceList(result.coaching_advice.priority_issues, 'issue')}
						</View>
					)}
					
					{result.coaching_advice.specific_tips && (
						<View style={styles.adviceSection}>
							<Text style={styles.adviceSectionTitle}>Specific Tips:</Text>
							{renderAdviceList(result.coaching_advice.specific_tips, 'tip')}
						</View>
					)}
					
					{result.coaching_advice.positive_feedback && (
						<View style={styles.adviceSection}>
							<Text style={styles.adviceSectionTitle}>What You're Doing Well:</Text>
							{renderAdviceList(result.coaching_advice.positive_feedback, 'positive')}
						</View>
					)}
					
					{result.coaching_advice.drill_suggestions && (
						<View style={styles.adviceSection}>
							<Text style={styles.adviceSectionTitle}>Practice Drills:</Text>
							{renderAdviceList(result.coaching_advice.drill_suggestions, 'drill')}
						</View>
					)}
				</Section>
			)}

			<Section title="Summary">
				<Text style={styles.dataText}>Frames: {result.frame_count}</Text>
				<Text style={styles.dataText}>FPS: {result.fps?.toFixed?.(2) ?? result.fps}</Text>
			</Section>

			{result.swing_phases && (
				<Section title="Swing Phases">
					{renderPairs(result.swing_phases)}
				</Section>
			)}

			{result.similarity_scores && (
				<Section title="Similarity vs Reference">
					{renderSimilarity(result.similarity_scores)}
				</Section>
			)}

			{result.message && (
				<Section title="Notes">
					<Text style={styles.dataText}>{result.message}</Text>
				</Section>
			)}
		</ScrollView>
	);
}

const styles = StyleSheet.create({
	container: {
		padding: 16,
	},
	emptyContainer: {
		flex: 1,
		alignItems: 'center',
		justifyContent: 'center',
	},
	emptyText: {
		fontSize: 16,
		color: '#666',
	},
	mainTitle: {
		fontSize: 24,
		fontWeight: '700',
		marginBottom: 8,
		color: '#333',
	},
	filename: {
		fontSize: 16,
		color: '#666',
		marginBottom: 20,
	},
	section: {
		marginTop: 20,
		backgroundColor: '#f8f9fa',
		padding: 16,
		borderRadius: 8,
	},
	sectionTitle: {
		fontSize: 18,
		fontWeight: '600',
		marginBottom: 12,
		color: '#333',
	},
	dataText: {
		fontSize: 14,
		color: '#555',
		marginBottom: 4,
	},
	summaryContainer: {
		backgroundColor: '#e3f2fd',
		padding: 12,
		borderRadius: 6,
		marginBottom: 16,
	},
	summaryText: {
		fontSize: 16,
		fontWeight: '500',
		color: '#1976d2',
		textAlign: 'center',
	},
	adviceSection: {
		marginBottom: 16,
	},
	adviceSectionTitle: {
		fontSize: 16,
		fontWeight: '600',
		marginBottom: 8,
		color: '#333',
	},
	adviceItem: {
		padding: 8,
		marginBottom: 6,
		borderRadius: 4,
		borderLeftWidth: 4,
	},
	advice_default: {
		backgroundColor: '#f5f5f5',
		borderLeftColor: '#9e9e9e',
	},
	advice_issue: {
		backgroundColor: '#ffebee',
		borderLeftColor: '#f44336',
	},
	advice_tip: {
		backgroundColor: '#e8f5e8',
		borderLeftColor: '#4caf50',
	},
	advice_positive: {
		backgroundColor: '#e3f2fd',
		borderLeftColor: '#2196f3',
	},
	advice_drill: {
		backgroundColor: '#fff3e0',
		borderLeftColor: '#ff9800',
	},
	adviceText: {
		fontSize: 14,
		color: '#333',
	},
}); 