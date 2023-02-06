import express from 'express';

// express server listneing on port 3000 which has one endpooint /test
const app = express();
app.get('/test', (req, res) => {
    res.send('Hello World!');
    });
app.listen(3000, () => console.log('Example app listening on port 3000!'));
