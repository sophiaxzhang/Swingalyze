import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import SwingScreen from '../screens/SwingScreen';
import ResultsScreen from '../screens/ResultsScreen';
import ReferenceScreen from '../screens/ReferenceScreen';

const Tab = createBottomTabNavigator();

export default function AppNavigator() {
	return (
		<Tab.Navigator screenOptions={{ headerShown: true }}>
			<Tab.Screen name="Analyze" component={SwingScreen} />
			<Tab.Screen name="Results" component={ResultsScreen} />
			<Tab.Screen name="Reference" component={ReferenceScreen} />
		</Tab.Navigator>
	);
} 