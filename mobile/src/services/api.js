import axios from 'axios';

const API_BASE_URL = 'http://192.168.68.73:8000'; //change later for production

//videoUri is file path pointing to selected vid
export const analyzeSwing = async (videoUri) => {
    const formData = new FormData();
    formData.append('file', {
        uri: videoUri,
        type: 'video/mp4',
        name: 'swing.mp4',
    });

    //send request with axios
    try {
        const response = await axios.post(`${API_BASE_URL}/analyze`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    } catch (error) {
        console.error('Analysis failed:', error);
        throw error;
    }
};